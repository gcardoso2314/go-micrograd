package datasets

import (
	"math"
	"math/rand"
)

func linSpace(start, end float64, numSamples int) []float64 {
	interval := (end - start) / float64(numSamples-1)
	result := make([]float64, numSamples)
	for i := 0; i < numSamples; i++ {
		result[i] = start + float64(i)*interval
	}
	return result
}

func MakeMoon(numSamples int, noise float64) ([][]float64, []float64) {
	numSamplesOut := numSamples / 2
	numSamplesIn := numSamples - numSamplesOut

	outerCircX := make([]float64, numSamplesOut)
	for i, v := range linSpace(0, math.Pi, numSamplesOut) {
		outerCircX[i] = math.Cos(v)
	}
	outerCircY := make([]float64, numSamplesOut)
	for i, v := range linSpace(0, math.Pi, numSamplesOut) {
		outerCircY[i] = math.Sin(v)
	}
	innerCircX := make([]float64, numSamplesIn)
	for i, v := range linSpace(0, math.Pi, numSamplesIn) {
		innerCircX[i] = 1 - math.Cos(v)
	}
	innerCircY := make([]float64, numSamplesIn)
	for i, v := range linSpace(0, math.Pi, numSamplesIn) {
		innerCircY[i] = 1 - math.Sin(v) - 0.5
	}

	X := make([][]float64, numSamples)
	y := make([]float64, numSamples)
	for i := 0; i < numSamples; i++ {
		if i < numSamplesOut {
			X[i] = []float64{
				rand.NormFloat64()*noise + outerCircX[i],
				rand.NormFloat64()*noise + outerCircY[i],
			}
			y[i] = 0.0
		} else {
			X[i] = []float64{
				rand.NormFloat64()*noise + innerCircX[i-numSamplesOut],
				rand.NormFloat64()*noise + innerCircY[i-numSamplesOut],
			}
			y[i] = 1.0
		}
	}

	return X, y

}
