package main

import (
	"log"
	"time"

	"gonum.org/v1/gonum/mat"
)

func main() {
	start := time.Now()

	// launchLeastSquares()
	launchKalmanFilter()
	// launchParticleFilter()

	elapsed := time.Since(start)
	log.Printf("Data %s", elapsed)
}

// FlatPlotPoints repesents data of some function values y = f(x)
type FlatPlotPoints struct {
	xs *mat.VecDense
	ys *mat.VecDense
}

func newFlatPlotPoints(size int) *FlatPlotPoints {
	xs := mat.NewVecDense(size, nil)
	ys := mat.NewVecDense(size, nil)
	return &FlatPlotPoints{xs, ys}
}

func linespace(begin, end float64, size int) *mat.VecDense {
	deltaT := (end - begin) / float64(size-1)
	t := mat.NewVecDense(size, nil)
	for i := 0; i < t.Len(); i++ {
		t.SetVec(i, float64(i)*deltaT)
	}
	return t
}
