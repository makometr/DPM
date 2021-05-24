package main

import (
	"fmt"
	"image/color"
	"math"

	odeint "github.com/Daniel-M/odeint/float64"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

var inputActions []func(n float64) float64 = []func(n float64) float64{
	func(n float64) float64 {
		return 2.5 * math.Sin(n)
	},
	func(n float64) float64 {
		return -2.0 * math.Cos(n)
	},
}

type kalmanFilter struct {
	deltaT      float64
	t           *mat.VecDense
	inputAction *FlatPlotPoints
}

func newKalmanFilter(t *mat.VecDense, fX func(number float64) float64, fY func(number float64) float64) *kalmanFilter {
	points := newFlatPlotPoints(t.Len())
	for i := 0; i < t.Len(); i++ {
		points.xs.SetVec(i, fX(t.AtVec(i)))
		points.ys.SetVec(i, fY(t.AtVec(i)))
	}

	return &kalmanFilter{t: t, inputAction: points, deltaT: t.AtVec(1) - t.AtVec(0)}
}

func (kf *kalmanFilter) launch() {
	odeResult := kf.calcODE()
	noiseODE := makeNormalNoise(odeResult)

	drawKalmanResult(odeResult, noiseODE, nil)

	FData := []float64{
		1, 0, kf.deltaT, 0,
		0, 1, 0, kf.deltaT,
		0, 0, 1, 0,
		0, 0, 0, 1,
	}
	F := mat.NewDense(4, 4, FData)

	GData := []float64{
		math.Pow(kf.deltaT, 2), 0,
		0, math.Pow(kf.deltaT, 2),
		kf.deltaT, 0,
		0, kf.deltaT,
	}
	G := mat.NewDense(4, 2, GData)

	Г := mat.NewDense(4, 2, GData)

	rawQ := []float64{
		1, 0,
		0, 1,
	}
	Q := mat.NewDense(2, 2, rawQ)
	fmt.Println(F)
	fmt.Println(G)
	fmt.Println(Г)
	fmt.Println(Q)
}

func (kf *kalmanFilter) calcODE() *FlatPlotPoints {
	var state []float64 = []float64{1, 1, 0, 0}

	index := 0
	odesys := func(x []float64, parameters []float64) []float64 {
		dxdt := make([]float64, len(x))
		t := kf.t.AtVec(index / 2) // костыль, т.к. эта функция на каждой итерации вызывается дважды
		dxdt[0] = x[2]
		dxdt[1] = x[3]
		dxdt[2] = inputActions[0](t)
		dxdt[3] = inputActions[1](t)

		index++
		return dxdt
	}
	system := odeint.NewSystem(state, nil, odesys)

	var integrator odeint.Midpoint
	err := integrator.Set(kf.deltaT, *system)
	if err != nil {
		panic(err)
	}

	result := newFlatPlotPoints(kf.t.Len())
	for i := 0; i < kf.t.Len(); i++ {
		result.xs.SetVec(i, state[0])
		result.ys.SetVec(i, state[1])
		// fmt.Println(float64(i)*integrator.StepSize(), state)

		state, err = integrator.Step()
		if err != nil {
			panic(err)
		}
	}
	return result
}

func makeNormalNoise(data *FlatPlotPoints) *FlatPlotPoints {
	size := data.xs.Len()
	result := newFlatPlotPoints(size)
	ND := distuv.Normal{Mu: 0, Sigma: 1}
	kf := 2.0

	for i := 0; i < size; i++ {
		result.xs.SetVec(i, data.xs.AtVec(i)+ND.Rand()*kf)
		result.ys.SetVec(i, data.ys.AtVec(i)+ND.Rand()*kf)
	}

	return result
}

func drawKalmanResult(trues, noisyTrues, filtered *FlatPlotPoints) {
	p := plot.New()
	pts := make(plotter.XYs, trues.xs.Len())

	for i := range pts {
		pts[i].X = trues.xs.AtVec(i)
		pts[i].Y = trues.ys.AtVec(i)
	}
	t, _ := plotter.NewLine(pts)
	t.LineStyle.Width = 2
	t.LineStyle.Color = color.RGBA{R: 255, A: 255}
	p.Legend.Add("Trues", t)

	for i := range pts {
		pts[i].X = noisyTrues.xs.AtVec(i)
		pts[i].Y = noisyTrues.ys.AtVec(i)
	}
	nt, _ := plotter.NewLine(pts)
	nt.LineStyle.Width = 1
	nt.LineStyle.Color = color.RGBA{G: 255, A: 255}
	p.Legend.Add("Noisy", nt)

	// for i := range pts {
	// 	pts[i].X = noisyTrues.xs.AtVec(i)
	// 	pts[i].Y = noisyTrues.ys.AtVec(i)
	// }
	// f, _ := plotter.NewLine(pts)
	// f.LineStyle.Width = 2
	// f.LineStyle.Color = color.RGBA{B: 255, A: 255}
	// p.Legend.Add("Filter", f)

	p.Title.Text = "МНК - пример построения"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"
	p.HideAxes()
	p.Add(nt, t)

	if err := p.Save(8*vg.Inch, 5*vg.Inch, "sas2.png"); err != nil {
		panic(err)
	}
}
