import sys
from pyspark import SparkConf, SparkContext
from math import sqrt

def filterDuplicates((vehicleID, partFrequency)):
    (part1, freq1) = partFrequency[0]
    (part2, freq2) = partFrequency[1]
    return part1 < part2
    
def makePairs((vehicleID, partFrequency)):
    (part1, freq1) = partFrequency[0]
    (part2, freq2) = partFrequency[1]
    return ((part1, part2), (freq1, freq2))

def computeCosineSimilarity(ratingPairs):
    numPairs = 0
    sum_xx = sum_yy = sum_xy = 0
    for ratingX, ratingY in ratingPairs:
        sum_xx += ratingX * ratingX
        sum_yy += ratingY * ratingY
        sum_xy += ratingX * ratingY
        numPairs += 1

    numerator = sum_xy
    denominator = sqrt(sum_xx) * sqrt(sum_yy)

    score = 0
    if (denominator):
        score = (numerator / (float(denominator)))

    return (score, numPairs)


conf = SparkConf().setMaster("local[*]").setAppName("partSimilarities")
sc = SparkContext(conf = conf)

print "\nLoading vehicle parts..."

data = sc.textFile("file:///SparkCourse/vehicle2parts.csv")
# test input data file format: vehicle ID, part ID, failureDate
# Map vehicles and parts to key / value pairs: ((vehicle ID, part ID), 1)
vp = data.map(lambda l: l.split(',')).map(lambda x: ((int(x[0]), int(x[1])), 1))

# input data file format: vehicle ID, part ID
# Map vehicles and parts to key / value pairs: ((vehicle ID, part ID), 1)
#vp = data.map(lambda l: l.split(',')).map(lambda x: ((x[0], x[1]), 1))
 
# Sum up the occurences of part replacements for each vehicle ID: ((vehicle ID, part ID), sum)
vpfreq = vp.reduceByKey(lambda x, y: (x + y))

# Rearrange to indicate part replacement freqency for each vehicle: (vehicle ID, (partID, frequency))
partfreq = vpfreq.map(lambda x: (x[0][0], (x[0][1], x[1]))) 

# Emit every pair of parts both replaced on the same vehicle.
# Self-join to find every combination.
joinedFreqencies = partfreq.join(partfreq)

# At this point our RDD consists of (vehicle ID, ((part ID, frequency), (part ID, frequency))
# Filter out duplicate pairs
uniqueJoinedFreqencies = joinedFreqencies.filter(filterDuplicates)

# Now key by (part1, part2) pairs to create ((part1, part2), (freqency1, frequency2))
partPairs = uniqueJoinedFreqencies.map(makePairs)
  
# Now collect all frequencies for each part pair and compute similarity
partPairFrequencies = partPairs.groupByKey()

# We now have ((part1, part2), (freq1, freq2), (freq1, freq2) ...)

#print "\nCount: " 
#print partPairFrequencies.count()

#for result in partPairFrequencies.takeSample(True,1000):
#    print result[0]
#    for res in result[1]:
#        print res
            
# Can now compute similarities.
partPairSimilarities = partPairFrequencies.mapValues(computeCosineSimilarity).cache()

# We now have ((part1, part2), (score, numpairs))
# Save the results if desired
partPairSimilarities.sortByKey()
partPairSimilarities.saveAsTextFile("part-sims")

# Extract similarities for the part we care about (arg) that have a similar vehicle co-occurrence failure frequency
if (len(sys.argv) > 1):

    scoreThreshold = 0.9
    # how many different vehicles must have cooccuring failures for the part
    coOccurenceThreshold = 5

    partID = int(sys.argv[1])

    # Filter for parts with this sim that are "similar" as defined by
    # our quality thresholds above
    filteredResults = partPairSimilarities.filter(lambda((pair,sim)): \
        (pair[0] == partID or pair[1] == partID) \
        and sim[0] > scoreThreshold and sim[1] >= coOccurenceThreshold)
        
    # Sort by quality score.
    results = filteredResults.map(lambda((pair,sim)): (sim, pair)).sortByKey(ascending = False).take(10)

    print "Top 10 similar parts for " + str(partID)
    for result in results:
        (sim, pair) = result
        # Display the similarity result that isn't the part we're looking at
        similarPartID = pair[0]
        if (similarPartID == partID):
            similarPartID = pair[1]
        print str(similarPartID) + "\tscore: " + str(sim[0]) + "\tstrength: " + str(sim[1])
