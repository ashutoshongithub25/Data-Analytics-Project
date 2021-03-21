# Natural Language Processing

# Importing the dataset
dataset_original = read.delim('data.txt', quote = '', stringsAsFactors = FALSE)
tail(colnames(dataset_original))
# Cleaning the texts
# install.packages('tm')
# install.packages('SnowballC')
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(dataset_original$headlines))
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords())
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)

# Creating the Bag of Words model
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.999)
dataset = as.data.frame(as.matrix(dtm))
dataset$outcome = dataset_original$outcome
dim(dataset)
tail(colnames(dataset))
head(colnames(dataset))
View(dataset)
# Importing the dataset
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[3:5]

# Encoding the target feature as factor
dataset$outcome = factor(dataset$outcome, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$outcome, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Random Forest Classification to the Training set
# install.packages('randomForest')
library(randomForest)
classifier = randomForest(x = training_set[-1475],
                          y = training_set$outcome,
                          ntree = 10)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-1475])

# Making the Confusion Matrix
cm = table(test_set[, 1475], y_pred)
View(cm)