import tensorflow as tf
from tensorflow import keras
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql.functions import udf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



def generate_dictionary(data):
    # Temp dictionary
    temp = {}
    word_dictionary = {}
    # Get all the text in the csv
    for text in data.values:
        lines = text[0]
        words = lines.split()
        for word in words:
            try:
                temp[word] = temp[word] + 1
            except KeyError:
                temp[word] = 0
    word_dictionary["<Unk>"] = 0
    word_dictionary["<Pad>"] = 1
    index = 2
    # Stop words. Same list as in P1 and P2
    stop_words = ["how", "after", "a", "with", "the", "in", "then", "out",
                  "which", "how's", "what", "when", "what's", "of",
                  "he", "she", "he's", "she's", "this", "that", "but",
                  "by", "at", "are", "and", "an", "as", "am", "i", "i've",
                  "any", "aren't", "be", "been", "being", "because", "can't",
                  "cannot", "could", "couldn't", "did", "didn't", "do", "does",
                  "doesn't", "don't", "doing", "for", "from", "has", "hasn't", "had",
                  "hadn't", "have", "haven't", "him", "her", "he'd", "he'll",
                  "his", "i'm", "i'll", "i'd", "if", "is", "isn't", "it", "it's",
                  "its", "let's", "or", "on", "other", "she'd", "she'll", "should",
                  "shouldn't", "so", "such", "that's", "they", "they're", "they've",
                  "their", "theirs", "this", "those", "to", "too", "very", "was",
                  "wasn't", "we", "we're", "we've", "we'll", "were", "weren't",
                  "when's", "where", "where's", "will", "who", "who's", "why",
                  "why's", "would", "wouldn't", "won't", "you", "your", "you'd",
                  "you'll", "you've", "yours", "me"]
    for key in temp.keys():
        # Add words to dictionary of they appear more than a 100 times and are not stop words
        if temp[key] > 100 and key not in stop_words:
            word_dictionary[key] = index
            index = index + 1
    return word_dictionary


def convert_data(data, w_dictionary):
    new_list = []
    # Get all the text in the csv
    for text in data.values:
        lines = text[0]
        words = lines.split()
        word_list = []
        for word in words:
            try:
                word_list.append(w_dictionary[word])
            except KeyError:
                word_list.append(0)
        new_list.append(np.asarray(word_list))
    return new_list


def convert_data2(data, w_dictionary):
    new_list = []
    # Get all the text in the csv
    for text in data.values:
        word_list = []
        words = text[0]
        for word in words:
            try:
                word_list.append(w_dictionary[word])
            except KeyError:
                word_list.append(0)
        new_list.append(np.asarray(word_list))
    return new_list


def getFullText(retweeted_status, quoted_status, text, extended_tweet):
    if retweeted_status is not None:
        if retweeted_status['truncated']:
            return retweeted_status['extended_tweet']['full_text'].lower()
        return retweeted_status['text'].lower()
    elif quoted_status is not None:
        if quoted_status['truncated']:
            return text + ":" + quoted_status['extended_tweet']['full_text']
        return quoted_status['text'].lower()
    else:
        if extended_tweet is None:
            return text
        return extended_tweet['full_text'].lower()

# File with training data in local system
file = "cleantextlabels7.csv"
# Get data
data = pd.read_csv(file, header=None, names=["text", "sentiment"])
# Separate labels from data
labels = data.sentiment
data = data.drop('sentiment', axis=1)
# Create word dictionary
word_dictionary = generate_dictionary(data)
# Convert text into indexes from word dictionary
converted_data = convert_data(data, word_dictionary)
# Placed converted data into a data frame
final_data = pd.DataFrame(np.array(converted_data).reshape(len(converted_data), 1), columns={"text"})
# Get datasets for both training and validating
train_data, test_data, train_labels, test_labels = train_test_split(final_data, labels, test_size=0.3)

print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

train_data = train_data.values.flatten()
test_data = test_data.values.flatten()
test_labels = test_labels.values.flatten()
train_labels = train_labels.values.flatten()

print(len(train_data[0]), len(train_data[1]))


train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_dictionary["<Pad>"],
                                                        padding='post',
                                                        maxlen=50)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_dictionary["<Pad>"],
                                                       padding='post',
                                                       maxlen=50)

print(len(train_data[0]), len(train_data[1]))
print(train_data[0])

x_val = train_data[:1000]
partial_x_train = train_data[1000:]

y_val = train_labels[:1000]
partial_y_train = train_labels[1000:]

# Model 1
model = keras.Sequential()
model.add(keras.layers.Embedding(len(word_dictionary), 64))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(64, activation=tf.nn.relu))
model.add(keras.layers.Dense(3, activation=tf.nn.softmax))

model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=50,
                    batch_size=200,
                    validation_data=(x_val, y_val),
                    verbose=1)

results = model.evaluate(test_data, test_labels)

# Model 2
model2 = keras.Sequential()
model2.add(keras.layers.Embedding(len(word_dictionary), 256))
model2.add(keras.layers.GlobalAveragePooling1D())
model2.add(keras.layers.Dense(256, activation=tf.nn.relu))
model2.add(keras.layers.Dense(128, activation=tf.nn.relu))
model2.add(keras.layers.Dense(64, activation=tf.nn.relu))
model2.add(keras.layers.Dense(32, activation=tf.nn.relu))
model2.add(keras.layers.Dense(3, activation=tf.nn.softmax))

model2.summary()

model2.compile(optimizer=tf.train.AdamOptimizer(),
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

history2 = model2.fit(partial_x_train,
                      partial_y_train,
                      epochs=30,
                      batch_size=150,
                      validation_data=(x_val, y_val),
                      verbose=1)

results2 = model2.evaluate(test_data, test_labels)

print("Results 1:")
print(results)
print("Results 2:")
print(results2)

# Get data that will be evaluated with models
initial_data = spark.read.json("/input/output_tweets.json")
# Function gets full tweet
full_text_extractor = udf(getFullText)
# Create new DataFrame with date column for when the data was created
tweets_df = initial_data.withColumn('full_text', full_text_extractor(initial_data['retweeted_status'], initial_data['quoted_status'], initial_data['text'], initial_data['extended_tweet']))
# Get only text data
text_df = tweets_df.select("full_text")
tokenizer = Tokenizer(inputCol="full_text", outputCol="words")
tokenized = tokenizer.transform(text_df)
remover = StopWordsRemover(inputCol="words", outputCol="base_words")
tweets_df2 = remover.transform(tokenized)
pd_tweets_df = tweets_df2.select("base_words").toPandas()
converted_tweets = convert_data2(pd_tweets_df, word_dictionary)
final_tweets_df = pd.DataFrame({"text": np.array(converted_tweets)})
final_tweets_df = final_tweets_df.values.flatten()
eval_data = keras.preprocessing.sequence.pad_sequences(final_tweets_df,
                                                       value=word_dictionary["<Pad>"],
                                                       padding='post',
                                                       maxlen=50)
model1_predictions = model.predict(eval_data, verbose=1)
model2_predictions = model2.predict(eval_data, verbose=1)
# This will be stored in local file system
output1 = open("prediction_model1.csv", "w")

for x in range(len(model1_predictions)):
    a = np.argmax(model1_predictions[x])
    b = np.argmax(model2_predictions[x])
    output1.write(str(a) + "," + str(b) + "\n")

# Get from local file system
model_df = spark.read.csv("file:///home/gretchen_bonilla/prediction_model1.csv")
merge_rdd = model_df.rdd.zip(tweets_df.select("full_text").rdd)
merge_df = spark.createDataFrame(merge_rdd)
output2 = open("pre_results.txt", "w")
output3 = open("graph_data.csv", 'w')

# Create view for using sql
merge_df.createOrReplaceTempView("merge")
# Flue results
flu_count1 = spark.sql("select _1._c0, _2.full_text from merge where _2.full_text like '%flu%' and _1._c0 == 1").count()
flu_count2 = spark.sql("select _1._c0, _2.full_text from merge where _2.full_text like '%flu%' and _1._c0 == 0").count()
flu_count3 = spark.sql("select _1._c1, _2.full_text from merge where _2.full_text like '%flu%' and _1._c1 == 1").count()
flu_count4 = spark.sql("select _1._c1, _2.full_text from merge where _2.full_text like '%flu%' and _1._c1 == 0").count()
flu_total = spark.sql("select _1._c0, _2.full_text from merge where _2.full_text like '%flu%'").count()
# Zika results
zika_count1 = spark.sql("select _1._c0, _2.full_text from merge where _2.full_text like '%zika%' and _1._c0 == 1").count()
zika_count2 = spark.sql("select _1._c0, _2.full_text from merge where _2.full_text like '%zika%' and _1._c0 == 0").count()
zika_count3 = spark.sql("select _1._c1, _2.full_text from merge where _2.full_text like '%zika%' and _1._c1 == 1").count()
zika_count4 = spark.sql("select _1._c1, _2.full_text from merge where _2.full_text like '%zika%' and _1._c1 == 0").count()
zika_total = spark.sql("select _1._c0, _2.full_text from merge where _2.full_text like '%zika%'").count()
# Diarrhea results
dia_count1 = spark.sql("select _1._c0, _2.full_text from merge where _2.full_text like '%diarrhea%' and _1._c0 == 1").count()
dia_count2 = spark.sql("select _1._c0, _2.full_text from merge where _2.full_text like '%diarrhea%' and _1._c0 == 0").count()
dia_count3 = spark.sql("select _1._c1, _2.full_text from merge where _2.full_text like '%diarrhea%' and _1._c1 == 1").count()
dia_count4 = spark.sql("select _1._c1, _2.full_text from merge where _2.full_text like '%diarrhea%' and _1._c1 == 0").count()
dia_total = spark.sql("select _1._c0, _2.full_text from merge where _2.full_text like '%diarrhea%'").count()
# Ebola results
ebola_count1 = spark.sql("select _1._c0, _2.full_text from merge where _2.full_text like '%ebola%' and _1._c0 == 1").count()
ebola_count2 = spark.sql("select _1._c0, _2.full_text from merge where _2.full_text like '%ebola%' and _1._c0 == 0").count()
ebola_count3 = spark.sql("select _1._c1, _2.full_text from merge where _2.full_text like '%ebola%' and _1._c1 == 1").count()
ebola_count4 = spark.sql("select _1._c1, _2.full_text from merge where _2.full_text like '%ebola%' and _1._c1 == 0").count()
ebola_total = spark.sql("select _1._c0, _2.full_text from merge where _2.full_text like '%ebola%'").count()
# Measles results
measles_count1 = spark.sql("select _1._c0, _2.full_text from merge where _2.full_text like '%measles%' and _1._c0 == 1").count()
measles_count2 = spark.sql("select _1._c0, _2.full_text from merge where _2.full_text like '%measles%' and _1._c0 == 0").count()
measles_count3 = spark.sql("select _1._c1, _2.full_text from merge where _2.full_text like '%measles%' and _1._c1 == 1").count()
measles_count4 = spark.sql("select _1._c1, _2.full_text from merge where _2.full_text like '%measles%' and _1._c1 == 0").count()
measles_total = spark.sql("select _1._c0, _2.full_text from merge where _2.full_text like '%measles%'").count()
# Write count results will be used for graphs
output3.write("keyword,model,p1,p2,total\n")
output3.write("flu,0," + str(flu_count1) + "," + str(flu_count2) + "," + str(flu_total) + "\n")
output3.write("flu,1," + str(flu_count3) + "," + str(flu_count4) + "," + str(flu_total) + "\n")
output3.write("zika,0," + str(zika_count1) + "," + str(zika_count2) + "," + str(zika_total) + "\n")
output3.write("zika,1," + str(zika_count3) + "," + str(zika_count4) + "," + str(zika_total) + "\n")
output3.write("diarrhea,0," + str(dia_count1) + "," + str(dia_count2) + "," + str(dia_total) + "\n")
output3.write("diarrhea,1," + str(dia_count3) + "," + str(dia_count4) + "," + str(dia_total) + "\n")
output3.write("ebola,0," + str(ebola_count1) + "," + str(ebola_count2) + "," + str(ebola_total) + "\n")
output3.write("ebola,1," + str(ebola_count3) + "," + str(ebola_count4) + "," + str(ebola_total) + "\n")
output3.write("measles,0," + str(measles_count1) + "," + str(measles_count2) + "," + str(measles_total) + "\n")
output3.write("measles,1," + str(measles_count3) + "," + str(measles_count4) + "," + str(measles_total) + "\n")
# Get all the results
results_df = spark.sql("select _1._c0, _1._c1, _2.full_text from merge")
# Write the results
for row in results_df.collect():
    output2.write("{ \"model_prediction1\":" + row._c0.encode("utf-8") + ", \"model_prediction2\":" + row._c1.encode("utf-8") + ", \"text\":" + row.full_text.encode("utf-8") + "}\n")



