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

	drawLeastSquares(basePoints, lsPoints)
}

func generateBaseData(size int) *FlatPlotPoints {
	x := make([]float64, size)
	y := make([]float64, size)
	for i := 0; i < size; i++ {
		x[i] = float64(i) / float64(size)
		y[i] = x[i] + rand.Float64() - 0.5
	}
	xs := mat.NewVecDense(size, x)
	ys := mat.NewVecDense(size, y)
	return &FlatPlotPoints{xs, ys}
}

// CalculateQuadroPolynomKfs calculates coefficients of quadratic polynomial for least square approximation method
func CalculateQuadroPolynomKfs(basePoints *FlatPlotPoints) *mat.Dense {
	// Values for quadratic polynomial
	size := basePoints.xs.Len()
	basicBasisValues := mat.NewDense(size, 3, nil)
	for i := 0; i < size; i++ {
		x := basePoints.xs.AtVec(i)
		basicBasisValues.Set(i, 0, x*x)
		basicBasisValues.Set(i, 1, x)
		basicBasisValues.Set(i, 2, 1)
	}

	kfs := calculateBasisKfs(basicBasisValues, basePoints.ys)
	return kfs
}

func calculateBasisKfs(basicBasisValues *mat.Dense, existedBasisValues *mat.VecDense) *mat.Dense {
	var a mat.Dense
	a.Product(basicBasisValues.T(), basicBasisValues) // ATA
	a.Inverse(&a)                                     // (ATA)-1
	var b mat.Dense
	b.Product(basicBasisValues.T(), existedBasisValues) // ATb
	b.Product(&a, &b)                                   // (ATA)-1 * ATb

	return &b
}

func drawLeastSquares(basePoints *FlatPlotPoints, lsPoints *FlatPlotPoints) {
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
