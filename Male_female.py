from sklearn.neighbors import KNeighborsClassifier

# [height, weight, shoe size]
inputData = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],[190, 90, 47], [175, 64, 39],[177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

inputLabel = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

classifier = KNeighborsClassifier()

# We train our classifier
classifier = classifier.fit(inputData, inputLabel)

# We test the classifier 
prediction = classifier.predict([[176, 75, 44]])

print(prediction) # ['male']
