from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow
import numpy
from sklearn import metrics
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plot
import seaborn as sb

model = tensorflow.keras.models.load_model(r'C:\Users\pa662\PycharmProjects\HGM\V11\gestpred.h5')
validation_generator = ImageDataGenerator()
validation_data_generator = validation_generator.flow_from_directory(r'C:\Users\pa662\PycharmProjects\HGM\V11\datasets\validation',
                                                        target_size=(64, 64),
                                                        batch_size=10,
                                                        color_mode='grayscale',
                                                        shuffle=False
                                                         )
test_steps_per_epoch = numpy.math.ceil(validation_data_generator.samples / validation_data_generator.batch_size)

predictions = model.predict_generator(validation_data_generator, steps=test_steps_per_epoch)
predicted_classes = numpy.argmax(predictions, axis=1)

#classification metrics
true_classes = validation_data_generator.classes
class_labels = list(validation_data_generator.class_indices.keys())
X = metrics.classification_report(true_classes, predicted_classes, target_names=class_labels)
print(X)

#Print confusion matrix
con_matrix = confusion_matrix(validation_data_generator.classes, predicted_classes)
normal_con_matrix = numpy.around(con_matrix.astype('float') / con_matrix.sum(axis=1)[:, numpy.newaxis], decimals=2)

Y = pd.DataFrame(con_matrix,
                columns = ["P.FIST", "P.ONE", "P.TWO", "P.THREE", "P.THUMBSUP", "P.FIVE", "P.SIX", "P.SEVEN"],
                index = ["A.FIST", "A.ONE", "A.TWO", "A.THREE", "A.THUMBSUP", "A.FIVE", "A.SIX", "A.SEVEN"],
                 )
print(Y)

#Print unnormalized confusion matrix
Z = pd.DataFrame(normal_con_matrix,
                columns = ["P.FIST", "P.ONE", "P.TWO", "P.THREE", "P.THUMBSUP", "P.FIVE", "P.SIX", "P.SEVEN"],
                index = ["A.FIST", "A.ONE", "A.TWO", "A.THREE", "A.THUMBSUP", "A.FIVE", "A.SIX", "A.SEVEN"],
                 )
print(Z)

# seaborn
#cmn = con_matrix.astype('float')
#con_matrix.sum(axis=1)[:, numpy.newaxis]
#fig, ax = plot.subplots(figsize=(10,10))
#gestname = ["FIST","ONE","TWO","THREE","THUMBSUP","FIVE","SIX","SEVEN"]
#sb.heatmap(cmn, annot=True, fmt='.2f', xticklabels = gestname, yticklabels= gestname)
#plot.ylabel('Actual')
#plot.xlabel('Predicted')
#plot.show(block=False)