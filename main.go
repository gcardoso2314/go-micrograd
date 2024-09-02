package main

import (
	"fmt"

	"github.com/gcardoso2314/go-micrograd/datasets"
	"github.com/gcardoso2314/go-micrograd/micrograd"
)

func main() {
	X, y := datasets.MakeMoon(100, 0.1)
	xs := micrograd.MatrixToValues(X)
	ys := micrograd.MatrixToValues([][]float64{y})[0]

	n, err := micrograd.NewMLP(2, []int{16, 16, 1})
	if err != nil {
		panic(err)
	}

	for i := 0; i < 100; i++ {
		ypred := make([]*micrograd.Value, len(xs))
		for i, x := range xs {
			out, err := n.Forward(x)
			if err != nil {
				panic(err)
			}
			ypred[i] = out
		}

		correctScores := 0.0
		for i := 0; i < len(ypred); i++ {
			if (ys[i].Data == 1.0) == (ypred[i].Data >= 0.5) {
				correctScores++
			}
		}
		accuracy := correctScores / float64(len(ypred))

		dataLoss := micrograd.BinaryCrossEntropy(ys, ypred)

		// apply regularization
		alpha := 1e-4
		regLoss := micrograd.NewLeafValue(0)
		for _, p := range n.Parameters() {
			regLoss = regLoss.Add(p.Mul(p))
		}
		regLoss = regLoss.Mul(alpha)
		totalLoss := dataLoss.Add(regLoss)

		n.ZeroGrad()
		totalLoss.Backward()

		learning_rate := 1.0 - 0.9*float64(i)/100
		for _, p := range n.Parameters() {
			p.Data -= learning_rate * p.Grad
		}

		fmt.Printf("\n%d: loss:%.4f | accuracy:%.2f\n", i, totalLoss.Data, accuracy)
	}
}
