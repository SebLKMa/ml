package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/mat"
)

func square(n float64) float64 {
	return n * n
}

func predictAsMatrix(x mat.Matrix, w mat.Matrix) mat.Matrix {
	// matrix Product
	var p mat.Dense
	p.Mul(x, w)

	// Print the result using the formatter.
	//fc := mat.Formatted(&p, mat.Prefix("    "), mat.Squeeze())
	//fmt.Printf("p = %v", fc)
	return &p
}

func predict(x mat.Matrix, w mat.Matrix) []float64 {
	// matrix Product
	var p mat.Dense
	p.Mul(x, w)

	pcols := mat.Col(nil, 0, &p)
	return pcols
}

// Performs np.average((predict(X, w) - Y) ** 2)
func loss(x mat.Matrix, y mat.Matrix, w mat.Matrix) float64 {
	predictions := predict(x, w)
	count := len(predictions)
	total := float64(0)
	for i, p := range predictions {
		//loss := square(p - y[i])
		loss := square(p - y.At(i, 0)) // At(row,col)
		total += loss
	}

	return total / float64(count)
}

// Performs 2 * np.matmul(X.T, (predict(X, w) - Y)) / X.shape[0]
func gradient(x mat.Matrix, y mat.Matrix, w mat.Matrix) mat.Matrix {
	predictions := predictAsMatrix(x, w) // predict(X, w)
	var sub mat.Dense
	sub.Sub(predictions, y) // (predict(X, w) - Y)

	var p mat.Dense
	p.Mul(x.T(), &sub)

	xshape := len(mat.Col(nil, 0, x))

	pCols := mat.Col(nil, 0, &p)
	divs := []float64{}
	for _, pc := range pCols {
		divs = append(divs, pc/float64(xshape))
	}

	muls := []float64{}
	for _, d := range divs {
		muls = append(muls, d*float64(2))
	}

	wshape := len(mat.Col(nil, 0, w))
	g := mat.NewDense(wshape, 1, muls)

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

// Trains and returns the optimum weights
func train(X mat.Matrix, Y mat.Matrix, iterations int, lr float64) mat.Matrix {
	w := mat.NewDense(3, 1, nil) // 3 columns, 1 row
	ok := true
	for i := 0; i < iterations; i++ {
		fmt.Printf("Iteration %d => Loss: %g\n", i, loss(X, Y, w))
		g := gradient(X, Y, w)
		matPrint(g)
		//w -= gradient(X, Y, w) * lr
		// cast to fix: cannot use decreaseGradient(g, lr, w) (type mat.Matrix) as type *mat.Dense in assignment: need type assertion
		w, ok = decreaseGradient(g, lr, w).(*mat.Dense)
		if !ok {
			log.Fatalln("Program error - decreaseGradient type not compatible with w")
		}
	}
	return w
}

func main() {

	x1, x2, x3, y, rowcount := loadInputsAndLabels("../../data/pizza3.txt")
	//inputs := append(append(append([]float64{}, x1...), x2...), x3...) // wrong dimension
	fmt.Printf("Input counts: %d\n", rowcount)
	inputs := []float64{}
	for i := 0; i < rowcount; i++ {
		inputs = append(inputs, x1[i])
		inputs = append(inputs, x2[i])
		inputs = append(inputs, x3[i])
	}

	// Create matrixes
	X := mat.NewDense(rowcount, 3, inputs)
	println("X:")
	matPrint(X)

	Y := mat.NewDense(rowcount, 1, y)
	println("Y:")
	matPrint(Y)

	w := mat.NewDense(3, 1, nil) // 3 columns, 1 row
	println("w:")
	matPrint(w)
	println("p:")
	p := predict(X, w)
	fmt.Printf("P: %+v\n", p)

	_ = train(X, Y, 10, float64(0.001))
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
