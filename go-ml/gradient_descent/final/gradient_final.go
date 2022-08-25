package main

import (
	"errors"
	"fmt"
	"log"
	"os"
	"strconv"
	"time"

	"github.com/go-gota/gota/dataframe"
)

// The mega-if statements in train() cannot scale with exponential combinations of cases.
// Gradient Descent is the faster, more precise and more general way to find the minimum loss.
// 3D plane Weight, Bias, Loss(x, y, z)
// w_gradient = 2 * np.average(X * (predict(X, w, b) - Y))
// b_gradient = 2 * np.average(predict(X, w, b) - Y)
func gradient(x []float64, y []float64, w float64, b float64) (wg float64, bg float64, err error) {
	lenX := len(x)
	if lenX != len(y) {
		return 0, 0, errors.New("Size of x and y must be the same size.")
	}

	w_total, b_total := float64(0), float64(0)

	// wg = 2 * np.average(X * (predict(X, w, b) - Y))
	// bg = 2 * np.average(predict(X, w, b) - Y)
	// This one loop could be faster
	for i, xv := range x {
		xa := []float64{xv} // e.g. of static typing vs python's dynamic typing
		p := predict(xa, w, b)
		w_xpy := xv * (p[0] - y[i])
		w_total += w_xpy
		b_xpy := p[0] - y[i]
		b_total += b_xpy
	}

	w_avg := w_total / float64(lenX)
	wg = float64(2) * w_avg
	b_avg := b_total / float64(lenX)
	bg = float64(2) * b_avg
	return
}

// Basically the y-intercept, not starting from origin 0,0 anymore
// Adding Bias can help to lower the error/loss
// Predicts pizzas(y) from reservations(x)
// y = x * w + b, where is the weight and b is the bias
func predict(x []float64, w float64, b float64) (predictions []float64) {
	for _, i := range x {
		predictions = append(predictions, i*w+b)
	}
	return
}

func square(n float64) float64 {
	return n * n
}

// loss refers to the error in prediction
// you get this error by comparing the actual X, Y from training data file vs result of predict(training data X, w)
// hence error = predict(X, w) -Y, which can result in -ve
// to avoid +ve loss, we square the prediction, error ** 2
// return np.average((predict(X, w) - Y) ** 2)
func loss(x []float64, y []float64, w float64, b float64) (float64, error) {
	lenX := len(x)
	if lenX != len(y) {
		return 0, errors.New("Size of x and y must be the same size.")
	}
	predictions := predict(x, w, b)
	//var losses []float64
	total := float64(0)
	for i, p := range predictions {
		loss := square(p - y[i])
		//losses = append(losses, square(p - y[i]))
		total += loss
	}

	return total / float64(lenX), nil
}

// lr refers to learning rate
// improve the weight until the loss is minimized
func train(x []float64, y []float64, iterations int, lr float64) (w float64, b float64, err error) {
	defer duration(timeIt("train() completed in"))
	/*
		w = 0
		for i in range(iterations):
			print("Iteration %4d => Loss: %.10f" % (i, loss(X, Y, w, 0)))
			w -= gradient(X, Y, w) * lr
		return w
	*/
	w, b = float64(0), float64(0)
	for i := 0; i < iterations; i++ {
		//current_loss, err := loss(x, y, w, b)
		if err != nil {
			return w, b, err
		}
		//fmt.Printf("Iteration %d w: %g => Loss: %g\n", i, w, current_loss)
		wg, bg, err := gradient(x, y, w, b)
		if err != nil {
			return wg, bg, err
		}
		w -= wg * lr
		b -= bg * lr
	}

	return w, b, nil
}

func main() {
	csv, err := os.Open("../../data/pizza.txt")
	if err != nil {
		log.Fatal(err)
	}

	df := dataframe.ReadCSV(csv, dataframe.WithDelimiter(','))
	//fmt.Println(df)

	var x []float64
	var y []float64
	records := df.Records()
	for _, r := range records {
		//fmt.Printf("%d: %s %s\n", i, r[0], r[1])
		f, err := strconv.ParseFloat(r[0], 64)
		if err != nil {
			log.Fatalln(err)
		}
		x = append(x, f)
		f, err = strconv.ParseFloat(r[1], 64)
		if err != nil {
			log.Fatalln(err)
		}
		y = append(y, f)
	}

	// Train the system, i.e. get the line for prediction
	w, b, err := train(x, y, 20000, float64(0.001))
	if err != nil {
		log.Fatalln(err)
	}
	fmt.Printf("w=%g b=%g\n", w, b)

	// Predict pizzas when reservations is 20
	input := []float64{float64(20)}
	p := predict(input, w, b)
	fmt.Printf("Prediction: x=%v => y=%v\n", input, p)

	// This Go version seems to perform much faster than Python's
}

func timeIt(msg string) (string, time.Time) {
	return msg, time.Now()
}

func duration(msg string, start time.Time) {
	//log.Printf("%v: %v\n", msg, time.Since(start))
	fmt.Printf("%v: %vns\n", msg, time.Since(start).Nanoseconds())
}
