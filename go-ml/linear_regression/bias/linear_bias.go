package main

import (
	"errors"
	"fmt"
	"log"
	"os"
	"strconv"

	"github.com/go-gota/gota/dataframe"
)

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
		return 0, errors.New("Size of x and y must be the same.")
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
	//w, b := float64(0), float64(0)
	for i := 0; i < iterations; i++ {
		current_loss, err := loss(x, y, w, b)
		if err != nil {
			return 0, 0, err
		}
		fmt.Printf("Iteration %d => Loss: %g\n", i, current_loss)
		/*
		   if loss(X, Y, w + lr, b) < current_loss:
		       w += lr
		   elif loss(X, Y, w - lr, b) < current_loss:
		       w -= lr
		   elif loss(X, Y, w, b + lr) < current_loss:
		       b += lr
		   elif loss(X, Y, w, b - lr) < current_loss:
		       b -= lr
		   else:
		       return w, b
		*/
		new_loss, err := loss(x, y, w+lr, b)
		if err != nil {
			return 0, 0, err
		}
		if new_loss < current_loss {
			w += lr
			continue
		}

		new_loss, err = loss(x, y, w-lr, b)
		if err != nil {
			return 0, 0, err
		}
		if new_loss < current_loss {
			w -= lr
			continue
		}

		new_loss, err = loss(x, y, w, b+lr)
		if err != nil {
			return 0, 0, err
		}
		if new_loss < current_loss {
			b += lr
			continue
		}

		new_loss, err = loss(x, y, w, b-lr)
		if err != nil {
			return 0, 0, err
		}
		if new_loss < current_loss {
			b -= lr
		} else {
			return w, b, nil
		}
	}

	return 0, 0, errors.New(fmt.Sprintf("Could not converge within %d iterations", iterations))
}

func main() {
	csv, err := os.Open("../pizza.txt")
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
	w, b, err := train(x, y, 10000, float64(0.01))
	if err != nil {
		log.Fatalln(err)
	}
	fmt.Printf("w=%g b=%g\n", w, b)

	//Predict pizzas when reservations is 20
	input := []float64{float64(20)}
	p := predict(input, w, b)
	fmt.Printf("Prediction: x=%v => y=%v\n", input, p)

}
