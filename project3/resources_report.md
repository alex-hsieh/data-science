# resources for the report

## loading and verifying the data shapes
    Shape of X_train: (7007, 28, 28, 3)
    Shape of y_train: (7007, 1)
    Shape of X_val: (1003, 28, 28, 3)
    Shape of y_val: (1003, 1)
    Shape of X_test: (2005, 28, 28, 3)
    Shape of y_test: (2005, 1)

It has 7,007 training images, 1,003 validation images, and 2,005 test images, all sized at 28x28 pixels with 3 color channels.


## The Class Distribution (Bar Chart)

### Look at that massive tower for Class 5! It contains over 4,000 images, while some of the other classes have fewer than 500.

    For your report: When answering "Are classes balanced?", you can definitively say no.

    When answering "Why does class imbalance matter?", you can reference our earlier discussion: a model could just predict Class 5 for every single patient and achieve a seemingly high "accuracy," completely failing to identify the other 6 conditions. This is exactly why we will need to look at Precision, Recall, and the F1-score later.