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

type flatPlotPoints struct {
	xs *mat.VecDense
	ys *mat.VecDense
}

func newFlatPlotPoints(size int) *flatPlotPoints {
	xs := mat.NewVecDense(size, nil)
	ys := mat.NewVecDense(size, nil)
	return &flatPlotPoints{xs, ys}
}

func main() {
	rand.Seed(1664)
	basePoints := generateBaseData(100)

	// Values for quadratic polynomial
	basicBasisValues := mat.NewDense(100, 3, nil)
	for i := 0; i < 100; i++ {
		x := basePoints.xs.AtVec(i)
		basicBasisValues.Set(i, 0, x*x)
		basicBasisValues.Set(i, 1, x)
		basicBasisValues.Set(i, 2, 1)
	}

	kfc := calculateBasisKfs(basicBasisValues, basePoints.ys)
	fmt.Printf("kfs: %0.4v\n", mat.Formatted(kfc, mat.Prefix("     ")))

	mnkPoints := newFlatPlotPoints(100)
	for i := 0; i < 100; i++ {
		x := basePoints.xs.AtVec(i)
		mnkPoints.xs.SetVec(i, x)
		mnkPoints.ys.SetVec(i, kfc.At(0, 0)*x*x+kfc.At(1, 0)*x+kfc.At(2, 0))
	}

	drawData(basePoints, mnkPoints)
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

func generateBaseData(size int) *flatPlotPoints {
	x := make([]float64, size)
	y := make([]float64, size)
	for i := 0; i < size; i++ {
		x[i] = float64(i) / 100
		y[i] = x[i] + rand.Float64() - 0.5
	}
	xs := mat.NewVecDense(size, x)
	ys := mat.NewVecDense(size, y)
	return &flatPlotPoints{xs, ys}
}

func drawData(points ...*flatPlotPoints) {
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
