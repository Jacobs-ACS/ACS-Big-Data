import sys
from pyspark import SparkConf, SparkContext
from math import sqrt

def filterDuplicates((partID, vehicleFrequency)):
    (vehicle1, freq1) = vehicleFrequency[0]
    (vehicle2, freq2) = vehicleFrequency[1]
    return vehicle1 < vehicle2
    
def makePairs((partID, vehicleFrequency)):
    (vehicle1, freq1) = vehicleFrequency[0]
    (vehicle2, freq2) = vehicleFrequency[1]
    return ((vehicle1, vehicle2), (freq1, freq2))

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


conf = SparkConf().setMaster("local[*]").setAppName("vehicleSimilarities")
sc = SparkContext(conf = conf)

print "\nLoading vehicle parts..."

data = sc.textFile("file:///SparkCourse/vehicleparts.csv")
# test input data file format: vehicle ID, part ID, failureDate
# Map vehicles and parts to key / value pairs: ((vehicle ID, part ID), 1)
vp = data.map(lambda l: l.split(',')).map(lambda x: ((int(x[0]), int(x[1])), 1))

# Sum up the occurences of part replacements for each vehicle ID: ((vehicle ID, part ID), sum)
vpfreq = vp.reduceByKey(lambda x, y: (x + y))

# Rearrange to indicate vehicle replacement freqency for each part: (part ID, (vehicle ID, frequency))
vfreq = vpfreq.map(lambda x: (x[0][1], (x[0][0], x[1]))) 

# Emit every pair of vehicles that had the same part replaced
# Self-join to find every combination.
joinedFreqencies = vfreq.join(vfreq)

# At this point our RDD consists of (part ID, ((vehicle ID, frequency), (vehicle ID, frequency))
# Filter out duplicate pairs
uniqueJoinedFreqencies = joinedFreqencies.filter(filterDuplicates)

# Now key by (vehicle1, vehicle2) pairs to create ((vehicle1, vehicle2), (freqency1, frequency2))
vehiclePairs = uniqueJoinedFreqencies.map(makePairs)
  
# Now collect all frequencies for each vehicle pair and compute similarity
vehiclePairFrequencies = vehiclePairs.groupByKey()

# We now have ((vehicle1, vehicle2), (freq1, freq2), (freq1, freq2) ...)

#print "\nCount: " 
#print vehiclePairFrequencies.count()

#for result in vehiclePairFrequencies.takeSample(True,1000):
#    print result[0]
#    for res in result[1]:
#        print res
            
# Can now compute similarities.
vehiclePairSimilarities = vehiclePairFrequencies.mapValues(computeCosineSimilarity).cache()

# We now have ((part1, part2), (score, numpairs))
# Save the results if desired
vehiclePairSimilarities.sortByKey()
vehiclePairSimilarities.saveAsTextFile("vehicle-sims")

# Extract similarities for the part we care about (arg) that have a similar vehicle co-occurrence failure frequency
if (len(sys.argv) > 1):

    scoreThreshold = 0.8
# how many different parts must be cooccuring failures on a vehicle
    coOccurenceThreshold = 10

    vehicleID = int(sys.argv[1])

    # Filter for parts with this sim that are "similar" as defined by
    # our quality thresholds above
    filteredResults = vehiclePairSimilarities.filter(lambda((pair,sim)): \
        (pair[0] == vehicleID or pair[1] == vehicleID) \
        and sim[0] > scoreThreshold and sim[1] >= coOccurenceThreshold)
        
    # Sort by quality score.
    results = filteredResults.map(lambda((pair,sim)): (sim, pair)).sortByKey(ascending = False).take(10)

    print "Top 10 similar vehicles for " + str(vehicleID)
    for result in results:
        (sim, pair) = result
        # Display the similarity result that isn't the vehicle we're looking at
        similarVehicleID = pair[0]
        if (similarVehicleID == vehicleID):
            similarVehicleID = pair[1]
        print str(similarVehicleID) + "\tscore: " + str(sim[0]) + "\tstrength: " + str(sim[1])
