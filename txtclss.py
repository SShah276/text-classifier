import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000) # Separate the training/testing data and create a 88000 word "vocabulary"

word_index = data.get_word_index() # Retrieves the integer mapping for the word

word_index = {k:(v+3) for k, v in word_index.items()} # Sets tags for all items
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value,key) for (key,value) in word_index.items()]) # Creates dictionary that switches key to integer and value to word

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=300)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=300)

def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

#print(decode_review(train_data[0]))

model = keras.Sequential()
model.add(keras.layers.Embedding(88000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, "relu"))
model.add(keras.layers.Dense(1, "sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

results = model.evaluate(test_data, test_labels)

print(results)

model.save("model.h5")


def review_encode(review):
    encoded_review = [1]

    for word in review:
        if word.lower() in word_index:
            encoded_review.append(word_index[word.lower()])
        else:
            encoded_review.append(2)

    return encoded_review

model = keras.models.load_model("model.h5")
with open("test.txt", encoding="utf-8") as f, open("predictions.txt", "w", encoding="utf-8") as output_file:
    for line in f.readlines():
        nine = line.replace(",", "").replace(".", "").replace("?", "").replace("!", "").replace(":","").strip().split(" ")
        encode = review_encode(nine)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250)
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])
        output_file.write(f"Original review: {line.strip()}\n")
        output_file.write(f"Encoded review: {encode.tolist()}\n")
        output_file.write(f"Prediction: {predict[0].tolist()}\n\n")

'''
test_review = test_data[0]
predict = model.predict([test_review])
print("Review: ")
print(decode_review(test_review))
print("Prediction: " + str(predict[0]))
print("Actual: " + str(test_labels[0]))
print(results)
'''