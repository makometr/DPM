package main

import (
	"fmt"
	"image/color"
	"math/rand"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
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
	deltaT := float64(8) / (100 - 1)
	t := mat.NewVecDense(100, nil)
	for i := 0; i < t.Len(); i++ {
		t.SetVec(i, float64(i)*deltaT)
	}

	filter := newKalmanFilter(t, inputActions[0], inputActions[1])
	filter.launch()
}

func launchLeastSquares() {
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

func drawData(basePoints *FlatPlotPoints, lsPoints *FlatPlotPoints) {
	p := plot.New()
	// descrs := []string{"Начальные точки", "Точки МНК-апппроксимации"}
	pts := make(plotter.XYs, basePoints.xs.Len())

	for i := range pts {
		pts[i].X = basePoints.xs.AtVec(i)
		pts[i].Y = basePoints.ys.AtVec(i)
	}
	lpPoints, _ := plotter.NewScatter(pts)
	lpPoints.Shape = draw.CircleGlyph{}

	for i := range pts {
		pts[i].X = lsPoints.xs.AtVec(i)
		pts[i].Y = lsPoints.ys.AtVec(i)
	}
	l, _ := plotter.NewLine(pts)
	l.LineStyle.Width = 2
	l.LineStyle.Color = color.RGBA{R: 255, A: 255}

	p.Title.Text = "МНК - пример построения"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"
	p.HideAxes()
	p.Legend.Add("Начальные точки", lpPoints)
	p.Legend.Add("МНК", l)
	p.Add(l, lpPoints)

	if err := p.Save(8*vg.Inch, 5*vg.Inch, "LeastSquaresExample.png"); err != nil {
		panic(err)
	}
}
