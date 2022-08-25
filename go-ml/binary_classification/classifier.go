package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

func sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}

// performs sigmoid on input matrix
func sigmoidMat(z mat.Matrix) mat.Matrix {
	zCols := mat.Col(nil, 0, z)
	sigmoids := []float64{}
	for _, z := range zCols {
		sigmoids = append(sigmoids, sigmoid(z))
	}

	zshape := len(mat.Col(nil, 0, z))
	results := mat.NewDense(zshape, 1, sigmoids)

	return results
}

func predict(x mat.Matrix, w mat.Matrix) []float64 {
	// matrix Product
	var p mat.Dense
	p.Mul(x, w)

	pcols := mat.Col(nil, 0, &p)
	return pcols
}

// See page 68, foward propagation is to predict
//def forward(X, w):
//    weighted_sum = np.matmul(X, w)
//    return sigmoid(weighted_sum)
func forward(x mat.Matrix, w mat.Matrix) mat.Matrix {
	predictions := predict(x, w)
	xshape := len(mat.Col(nil, 0, x))
	weightedSum := mat.NewDense(xshape, 1, predictions)
	return sigmoidMat(weightedSum)
}

// Note that the labels used to train the classifier are either 0 or 1.
// Classify is a form of prediction, in this case, to either 0 or 1 (rounded), using the weightage.
//def classify(X, w):
//    return np.round(forward(X, w))
func classify(x mat.Matrix, w mat.Matrix) mat.Matrix {
	forwards := forward(x, w)
	fCols := mat.Col(nil, 0, forwards)
	roundeds := []float64{}
	for _, f := range fCols {
		roundeds = append(roundeds, math.Round(f)) // 0 < f < 1, Round() will round value to 0 or 1
	}

	xshape := len(mat.Col(nil, 0, x))
	classifieds := mat.NewDense(xshape, 1, roundeds)

	return classifieds
}

// See page 70 log loss formula
//def loss(X, Y, w):
//    y_hat = forward(X, w)
//    first_term = Y * np.log(y_hat)
//    second_term = (1 - Y) * np.log(1 - y_hat)
//    return -np.average(first_term + second_term)
func loss(x mat.Matrix, y mat.Matrix, w mat.Matrix) float64 {
	yHat := forward(x, w) // yHat contains the predictions
	yhCols := mat.Col(nil, 0, yHat)
	yCols := mat.Col(nil, 0, y)
	firstPartCols := []float64{}
	for i, yValue := range yCols {
		partValue := yValue * math.Log(yhCols[i])
		firstPartCols = append(firstPartCols, partValue)
	}
	secondPartCols := []float64{}
	for i, yValue := range yCols {
		partValue := (1 - yValue) * math.Log(1-yhCols[i])
		secondPartCols = append(secondPartCols, partValue)
	}
	sumParts := []float64{}
	for i, second := range secondPartCols {
		sumParts = append(sumParts, firstPartCols[i]+second)
	}
	average := stat.Mean(sumParts, nil)

	return -average
}

// The goal of gradient descent is to move downhill
//def gradient(X, Y, w):
//    return np.matmul(X.T, (forward(X, w) - Y)) / X.shape[0]
// Performs 2 * np.matmul(X.T, (predict(X, w) - Y)) / X.shape[0]
func gradient(x mat.Matrix, y mat.Matrix, w mat.Matrix) mat.Matrix {
	forwards := forward(x, w) // forward(X, w)
	var sub mat.Dense
	sub.Sub(forwards, y) // (forward(X, w) - Y)

	var p mat.Dense
	p.Mul(x.T(), &sub) // np.matmul(X.T, (forward(X, w) - Y))

	xshape := len(mat.Col(nil, 0, x)) // X.shape[0]

	pCols := mat.Col(nil, 0, &p)
	divs := []float64{}
	for _, pc := range pCols {
		divs = append(divs, pc/float64(xshape)) // np.matmul(X.T, (forward(X, w) - Y)) / X.shape[0]
	}

	wshape := len(mat.Col(nil, 0, w))
	g := mat.NewDense(wshape, 1, divs)

	return g
}

// Performs w -= gradient(X, Y, w) * lr
func decreaseGradient(g mat.Matrix, lr float64, w mat.Matrix) mat.Matrix {
	gCols := mat.Col(nil, 0, g)
	glr := []float64{}
	for _, gc := range gCols {
		glr = append(glr, gc*lr)
	}

	wCols := mat.Col(nil, 0, w)
	nw := []float64{}
	for i, wc := range wCols {
		wc -= glr[i]
		nw = append(nw, wc)
	}

	wshape := len(mat.Col(nil, 0, w))
	ng := mat.NewDense(wshape, 1, nw)
	return ng
}

// Returns the weightage after training
//def train(X, Y, iterations, lr):
//    w = np.zeros((X.shape[1], 1))
//    for i in range(iterations):
//        print("Iteration %4d => Loss: %.20f" % (i, loss(X, Y, w)))
//        w -= gradient(X, Y, w) * lr
//    return w
func train(X mat.Matrix, Y mat.Matrix, iterations int, lr float64) mat.Matrix {
	w := mat.NewDense(4, 1, nil) // 1(bias) + 3 columns, 1 row
	ok := true
	for i := 0; i < iterations; i++ {
		fmt.Printf("Iteration %d => Loss: %g\n", i, loss(X, Y, w))
		g := gradient(X, Y, w)
		//matPrint(g)
		//w -= gradient(X, Y, w) * lr
		// cast to fix: cannot use decreaseGradient(g, lr, w) (type mat.Matrix) as type *mat.Dense in assignment: need type assertion
		w, ok = decreaseGradient(g, lr, w).(*mat.Dense)
		if !ok {
			log.Fatalln("Program error - decreaseGradient result is not compatible with w")
		}
	}
	return w
}

// Test the classification of inputs with weightage against the labels
//def test(X, Y, w):
//    total_examples = X.shape[0]
//    correct_results = np.sum(classify(X, w) == Y)
//    success_percent = correct_results * 100 / total_examples
//    print("\nSuccess: %d/%d (%.2f%%)" % (correct_results, total_examples, success_percent))
func test(x mat.Matrix, y mat.Matrix, w mat.Matrix) {
	totalExamples := len(mat.Col(nil, 0, x)) // X.shape[0]
	classifieds := classify(x, w)
	classifiedCols := mat.Col(nil, 0, classifieds)
	correctResults := 0
	yCols := mat.Col(nil, 0, y)
	for i, y := range yCols {
		if classifiedCols[i] == y {
			correctResults++
		}
	}
	successPercent := float64(correctResults) * float64(100) / float64(totalExamples)
	fmt.Printf("\nSuccess: %d/%d (%.2f%%)\n", correctResults, totalExamples, successPercent)
}

func main() {
	x1, x2, x3, y, rowcount := loadInputsAndLabels("../data/police.txt")

	fmt.Printf("Input counts: %d\n", rowcount)

	// The inputs for creating the X matrix with bias in first column
	inputs := []float64{}
	for i := 0; i < rowcount; i++ {
		inputs = append(inputs, float64(1)) // the x0 for bias
		inputs = append(inputs, x1[i])
		inputs = append(inputs, x2[i])
		inputs = append(inputs, x3[i])
	}

	// Create X matrix with additional x0 as bias
	X := mat.NewDense(rowcount, 4, inputs)
	//println("X:")
	//matPrint(X)

	Y := mat.NewDense(rowcount, 1, y)
	//println("Y:")
	//matPrint(Y)

	w := train(X, Y, 10000, float64(0.001))

	println("\nThe transposed Weights: [[ Bias, Reservations, Temperature, Tourists]]")
	matPrint(w.T())

	// Test the classification with the "trained" weightage
	test(X, Y, w)

	// This Go version is faster than the Python (start both at the same time to compare)
}

func matPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

func loadInputsAndLabels(filename string) (x1 []float64, x2 []float64, x3 []float64, y []float64, inputCount int) {
	csvFile, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}

	defer csvFile.Close()

	reader := csv.NewReader(csvFile)

	reader.Comma = '\t' // Use tab-delimited instead of comma <---- here!

	reader.FieldsPerRecord = -1

	csvData, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	for i, line := range csvData {
		//fmt.Printf("%d: %+v\n", i, line)
		// skip header line
		if i == 0 {
			continue
		}

		cols := strings.Fields(line[0])
		if len(cols) != 4 {
			log.Fatalf("Expecting 4 columns. Got %d\n", len(cols))
		}
		x1 = append(x1, discardParse64Error(cols[0]))
		x2 = append(x2, discardParse64Error(cols[1]))
		x3 = append(x3, discardParse64Error(cols[2]))
		y = append(y, discardParse64Error(cols[3]))
		/*
			for _, c := range cols {
				f, err := strconv.ParseFloat(c, 64)
				if err != nil {
					log.Fatalln(err)
				}
				fmt.Printf("%g ", f)
			}
			fmt.Println()
		*/
	}

	fmt.Println("Sanity check...")
	j := 0
	for i, line := range csvData {
		fmt.Printf("%d: %+v\n", i, line)
		// skip header line
		if i == 0 {
			continue
		}
		fmt.Printf("%d:  %g ", i, x1[j])
		fmt.Printf("%g ", x2[j])
		fmt.Printf("%g ", x3[j])
		fmt.Printf("%g\n", y[j])
		j++
	}
	inputCount = j
	return
}

func discardParse64Error(s string) (f float64) {
	f, _ = strconv.ParseFloat(s, 64)
	return
}
