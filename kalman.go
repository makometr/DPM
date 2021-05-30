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

func launchKalmanFilter() {
	t := linespace(0, 8, 100)
	kf := newKalmanFilter(t, inputActions[0], inputActions[1])
	odeResult := kf.calcODE()
	noiseODE := makeNormalNoise(odeResult)

	R := mat.NewDense(2, 2, []float64{
		4, 0,
		0, 4,
	})
	H := mat.NewDense(2, 4, []float64{
		1, 0, 0, 0,
		0, 1, 0, 0,
	})
	Xi1 := mat.NewDense(4, 1, []float64{
		1,
		1,
		0,
		0,
	})
	Pi1 := mat.NewDense(4, 4, []float64{
		0.1, 0, 0, 0,
		0, 0.1, 0, 0,
		0, 0, 0.1, 0,
		0, 0, 0, 0.1,
	})
	// ti1 := 0

	F := mat.NewDense(4, 4, []float64{
		1, 0, kf.deltaT, 0,
		0, 1, 0, kf.deltaT,
		0, 0, 1, 0,
		0, 0, 0, 1,
	})

	G := mat.NewDense(4, 2, []float64{
		math.Pow(kf.deltaT, 2), 0,
		0, math.Pow(kf.deltaT, 2),
		kf.deltaT, 0,
		0, kf.deltaT,
	})

	Г := mat.NewDense(4, 2, []float64{
		math.Pow(kf.deltaT, 2), 0,
		0, math.Pow(kf.deltaT, 2),
		kf.deltaT, 0,
		0, kf.deltaT,
	})

	Q := mat.NewDense(2, 2, []float64{
		1, 0,
		0, 1,
	})

	x := make([]float64, 100)
	x[0] = 1
	y := make([]float64, 100)
	y[0] = 1

	for ti := 1; ti < 100; ti++ {
		fmt.Println("Step ", ti)
		U := mat.NewVecDense(2, []float64{
			kf.inputAction.xs.AtVec(ti),
			kf.inputAction.ys.AtVec(ti),
		})
		Z := mat.NewVecDense(2, []float64{
			noiseODE.xs.AtVec(ti),
			noiseODE.ys.AtVec(ti),
		})
		Xi, Pi := doWork(Xi1, U, F, G, Pi1, Q, Г, Z, H, R)
		x[ti] = Xi.At(0, 0)
		y[ti] = Xi.At(1, 0)
		Xi1.Copy(Xi)
		Pi1.Copy(Pi)
	}

	drawKalmanResult(odeResult, noiseODE, &FlatPlotPoints{xs: mat.NewVecDense(100, x), ys: mat.NewVecDense(100, y)})
}

func doWork(X, U, F, G, P, Q, Г, Z, H, R mat.Matrix) (*mat.Dense, *mat.Dense) {
	var XPrior, XPlhs, XPrhs mat.Dense
	XPlhs.Product(F, X)
	XPrhs.Product(G, U)
	XPrior.Add(&XPlhs, &XPrhs)

	var Pprior, ppriorlhs, ppriorrhs mat.Dense
	ppriorlhs.Product(F, P, F.T())
	ppriorrhs.Product(Г, Q, Г.T())
	Pprior.Add(&ppriorlhs, &ppriorrhs)

	var y mat.Dense
	y.Product(H, &XPrior)
	y.Sub(Z, &y)

	var S mat.Dense
	S.Product(H, &Pprior, H.T())
	S.Add(&S, R)

	var K, Sinv mat.Dense
	Sinv.Inverse(&S)
	K.Product(&Pprior, H.T(), &Sinv)

	var XPost, xpostrhs mat.Dense
	xpostrhs.Product(&K, &y)
	XPost.Add(&XPrior, &xpostrhs)

	var PPost mat.Dense
	size, _ := X.Dims()
	eye := mat.NewDiagDense(size, nil)
	for i := 0; i < size; i++ {
		eye.SetDiag(i, 1)
	}
	PPost.Product(&K, H)
	PPost.Sub(eye, &PPost)
	PPost.Product(&PPost, &Pprior)

	return &XPost, &PPost
}

func (kf *kalmanFilter) calcODE() *FlatPlotPoints {
	var state []float64 = []float64{1, 1, 0, 0}

	index := 0
	odesys := func(x []float64, parameters []float64) []float64 {
		dxdt := make([]float64, len(x))
		dxdt[0] = x[2]
		dxdt[1] = x[3]
		// костыль, т.к. эта функция на каждой итерации вызывается дважды
		dxdt[2] = kf.inputAction.xs.AtVec(index / 2)
		dxdt[3] = kf.inputAction.ys.AtVec(index / 2)

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

		state, err = integrator.Step()
		if err != nil {
			panic(err)
		}
	}
	return result
}

func makeNormalNoise(data *FlatPlotPoints) *FlatPlotPoints {
	// size := data.xs.Len()
	// result := newFlatPlotPoints(size)

	// for i := 0; i < size; i++ {
	// 	var sign float64
	// 	if i%2 == 1 {
	// 		sign = 1
	// 	} else {
	// 		sign = -1
	// 	}
	// 	result.xs.SetVec(i, data.xs.AtVec(i)+(-1)*sign)
	// 	result.ys.SetVec(i, data.ys.AtVec(i)+(-1)*sign)
	// }

	// return result
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

	for i := range pts {
		pts[i].X = filtered.xs.AtVec(i)
		pts[i].Y = filtered.ys.AtVec(i)
	}
	f, _ := plotter.NewLine(pts)
	f.LineStyle.Width = 2
	f.LineStyle.Color = color.RGBA{B: 255, A: 255}
	p.Legend.Add("Filter", f)

	p.Title.Text = "МНК - пример построения"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"
	// p.HideAxes()
	p.Add(nt, t, f)

	if err := p.Save(8*vg.Inch, 5*vg.Inch, "KalmanExample.png"); err != nil {
		panic(err)
	}
}
