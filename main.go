package main

import (
	"fmt"
	"math/rand"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

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

func main() {
	rand.Seed(1664)
	sampleSize := 100
	basePoints := generateBaseData(sampleSize)

	kfc := CalculateQuadroPolynomKfs(basePoints)
	fmt.Printf("kfs: %0.4v\n", mat.Formatted(kfc, mat.Prefix("     ")))

	lsPoints := newFlatPlotPoints(sampleSize) // least square points
	for i := 0; i < lsPoints.xs.Len(); i++ {
		x := basePoints.xs.AtVec(i)
		lsPoints.xs.SetVec(i, x)
		lsPoints.ys.SetVec(i, kfc.At(0, 0)*x*x+kfc.At(1, 0)*x+kfc.At(2, 0))
	}

	drawData(basePoints, lsPoints)
}

func generateBaseData(size int) *FlatPlotPoints {
	x := make([]float64, size)
	y := make([]float64, size)
	for i := 0; i < size; i++ {
		x[i] = float64(i) / 100
		y[i] = x[i] + rand.Float64() - 0.5
	}
	xs := mat.NewVecDense(size, x)
	ys := mat.NewVecDense(size, y)
	return &FlatPlotPoints{xs, ys}
}

func drawData(points ...*FlatPlotPoints) {
	p := plot.New()
	descrs := []string{"Начальные точки", "Точки МНК-апппроксимации"}
	for pointsIndex := 0; pointsIndex < len(points); pointsIndex++ {
		pts := make(plotter.XYs, points[pointsIndex].xs.Len())
		for i := range pts {
			pts[i].X = points[pointsIndex].xs.AtVec(i)
			pts[i].Y = points[pointsIndex].ys.AtVec(i)
		}
		plotutil.AddLines(p, descrs[pointsIndex], pts)
	}
	p.Title.Text = "МНК - пример построения"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"
	p.HideAxes()

	if err := p.Save(8*vg.Inch, 5*vg.Inch, "LeastSquaresExample.png"); err != nil {
		panic(err)
	}
}