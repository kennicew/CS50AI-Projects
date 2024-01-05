import csv
from pickle import NONE
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """

    evidence = []
    labels = []

    MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    with open(filename, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            print('Loading data from csv file...')
            rows = 0

            for row in csvreader:
                if rows > 0:
                    row_evidence = []

                    row_evidence.append(int(row[0]))
                    row_evidence.append(float(row[1]))
                    row_evidence.append(int(row[2]))
                    row_evidence.append(float(row[3]))
                    row_evidence.append(int(row[4]))
                    row_evidence.append(float(row[5]))
                    row_evidence.append(float(row[6]))
                    row_evidence.append(float(row[7]))
                    row_evidence.append(float(row[8]))
                    row_evidence.append(float(row[9]))
                    row_evidence.append(MONTHS.index(row[10]))
                    row_evidence.append(int(row[11]))
                    row_evidence.append(int(row[12]))
                    row_evidence.append(int(row[13]))
                    row_evidence.append(int(row[14]))
                    row_evidence.append(1 if row[15]=='Returning_Visitor' else 0)
                    row_evidence.append(1 if row[16]=='TRUE' else 0)

                    evidence.append(row_evidence)
                    labels.append(1 if row[17]=='TRUE' else 0)
            
                rows += 1
            
            return (evidence, labels)

    


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model



def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    positives = 0
    negatives = 0
    true_positives = 0
    true_negatives = 0

    for i in range(len(labels)):
        label = labels[i]
        prediction = predictions[i]

        if label == 1:
            positives += 1
            if prediction == label:
                true_positives += 1
        else:
            negatives += 1
            if prediction == label:
                true_negatives += 1

    sensitivity = true_positives / positives
    specificity = true_negatives / negatives

    return(sensitivity, specificity)


if __name__ == "__main__":
    main()
