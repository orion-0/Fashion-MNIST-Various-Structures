# Fashion-MNIST
This python code uses a pytorch implementation of use of Convolutional Network on Fashion-MNIST dataset using Pytorch. 

## Dataset
Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. Zalando intends Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.

Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255. The training and test data sets have 785 columns. The first column consists of the class labels (see above), and represents the article of clothing. The rest of the columns contain the pixel-values of the associated image.

## Labels
Each training and test example is assigned to one of the following labels:

0 T-shirt/top
1 Trouser
2 Pullover
3 Dress
4 Coat
5 Sandal
6 Shirt
7 Sneaker
8 Bag
9 Ankle boot 

## Model
The Model consist of convolutional Layers with relu activation functions and Fully Connected layers.
Following is structure of the Convoltional Model Used 

```
Net(
  (conv1): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(20, 50, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=800, out_features=500, bias=True)
  (fc2): Linear(in_features=500, out_features=10, bias=True)
)
```

## Gradient Descent Used
### Stochastic gradient descent
Stochastic gradient descent (SGD) in contrast performs a parameter update for each training example x(i) and label y(i):
```
θ=θ−η⋅∇θJ(θ;x(i);y(i)).
```
Batch gradient descent performs redundant computations for large datasets, as it recomputes gradients for similar examples before each parameter update. SGD does away with this redundancy by performing one update at a time. It is therefore usually much faster and can also be used to learn online. 
SGD performs frequent updates with a high variance that cause the objective function to fluctuate heavily. Hence SGD was used as the base algorithm.
```
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```
## Loss
### Cross Entropy Loss
Cross-entropy loss, or log loss, measures the performance of a classification model whose output is a probability value between 0 and 1. Cross-entropy loss increases as the predicted probability diverges from the actual label. However they work fairly well with Image datasets.
```
criterion = nn.CrossEntropyLoss()
```

## Accuracy
The max Achieved Accuracy is found to be 92% and the average accuracy is found to be 87.5% after 25 epochs on a GPU with GTX 1050 Configuration with a batch size of 75.
