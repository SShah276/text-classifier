# IMDB Movie Review Sentiment Classifier 🎬🤖

### Dataset: IMDB Movie Reviews, pre-tokenized by TensorFlow 
--> https://www.tensorflow.org/tutorials/keras/text_classification_with_hub

 Using TensorFlow and Keras, I built and trained a Neural Network to classify the sentiment of movie reviews.
 The model predicts whether a review is positive or negative.

To test this model on outside data, I used Google reviews on the movie Avengers: Endgame, 
and output the results into predictions.txt  
 
Model Architecture:  
Input: Embedding Layer,  
Hidden: GlobalAveragePoolingID,  
Hidden: Dense(ReLU),   
Output Layer: Dense(Sigmoid)

Training:
- Trained with binary cross-entropy loss and the Adam optimizer
- Ran for 40 epochs with batch size 512
- Validation split: 10,000 samples for validation during training