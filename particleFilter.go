package main

import (
	"fmt"
	"image/color"
	"io/ioutil"
	"math"
	"math/rand"

	"github.com/icza/mjpeg"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg/draw"
)

func origF(elem float64) float64 {
	return math.Exp(elem)
}

func fFmodel(x float64, θ mat.Vector) float64 {
	c0, c1, c2, c3 := θ.AtVec(0), θ.AtVec(1), θ.AtVec(2), θ.AtVec(3)
	return c0*math.Exp(c1*x+c2) + c3
}

func calcDelta(t, Z mat.Vector, θ mat.Vector) (result float64) {
	for i := 0; i < Z.Len(); i++ {
		result += math.Pow(math.Abs(Z.AtVec(i)-fFmodel(t.AtVec(i), θ)), 2)
	}
	return result
}

type filterState struct {
	θs      *mat.Dense
	weights *mat.VecDense
}

func newFilterState(N int) *filterState {
	cfsNumber := 4
	θs := make([]float64, N*cfsNumber)
	for i := 0; i < N; i++ {
		θs[i*cfsNumber+0] = randF(0.2, 1.1)
		θs[i*cfsNumber+1] = randF(0.2, 1.1)
		θs[i*cfsNumber+2] = randF(0.2, 1.1)
		θs[i*cfsNumber+3] = randF(0.2, 1.1)
	}

	weights := make([]float64, N)
	for i := 0; i < N; i++ {
		weights[i] = 1.0 / float64(N)
	}

	return &filterState{θs: mat.NewDense(N, 4, θs), weights: mat.NewVecDense(N, weights)}
}

func (fs *filterState) updateWeights(t, Z mat.Vector) {
	for i := 0; i < fs.weights.Len(); i++ {
		deltaValue := calcDelta(t, Z, fs.θs.RowView(i))
		fs.weights.SetVec(i, 1/deltaValue)
	}
	fs.normalizeWeights()
}

func (fs *filterState) normalizeWeights() {
	sum := mat.Sum(fs.weights)
	for i := 0; i < fs.weights.Len(); i++ {
		fs.weights.SetVec(i, fs.weights.AtVec(i)/sum)
	}
}

func (fs *filterState) updateθs() {
	newθs := mat.NewDense(fs.θs.RawMatrix().Rows, fs.θs.RawMatrix().Cols, nil)
	newWeights := mat.NewVecDense(fs.weights.Len(), nil)

	index := rand.Intn(fs.θs.RawMatrix().Rows)
	betta := 0.0

	UD := distuv.Uniform{}
	UD.Min = 0
	UD.Max = mat.Max(fs.weights) * 2
	for i := 0; i < fs.θs.RawMatrix().Rows; i++ {
		betta += UD.Rand()
		for betta > fs.weights.AtVec(index) {
			betta -= fs.weights.AtVec(index)
			index = (index + 1) % 20
		}
		newθs.SetRow(i, fs.θs.RawRowView(index))
		newWeights.SetVec(i, fs.weights.AtVec(index))
	}

	fs.θs = newθs
	fs.weights = newWeights
	// запоминаем индексы, если больше одного, то шифтим
}

func launchParticleFilter() {
	N := 20
	deltaT := float64(2) / (20 - 1)
	X := mat.NewVecDense(20, nil)
	for i := 0; i < X.Len(); i++ {
		X.SetVec(i, float64(i)*deltaT)
	}

	filterState := newFilterState(N)
	Z := getNoiseData(X, origF)

	for i := 0; i < N; i++ {
		filterState.updateWeights(X.SliceVec(0, i+1), Z.SliceVec(0, i+1))
		// fmt.Printf("weights: %0.4v\n", mat.Formatted(filterState.weights, mat.Prefix("         ")))

		if (i+1)%4 == 0 {
			filterState.updateθs()
		}
		drawparticleFilerStep("pf"+fmt.Sprint(i), X, filterState, X.SliceVec(0, i+1), Z.SliceVec(0, i+1))
	}

	checkErr := func(err error) {
		if err != nil {
			panic(err)
		}
	}

	aw, err := mjpeg.New("pf.avi", 1000, 800, 2)
	checkErr(err)

	for i := 0; i < N; i++ {
		data, err := ioutil.ReadFile(fmt.Sprintf("pf"+"%d.jpg", i))
		checkErr(err)
		checkErr(aw.AddFrame(data))
	}

	checkErr(aw.Close())
}

func getNoiseData(xs mat.Vector, realFunction func(float64) float64) *mat.VecDense {
	ND := distuv.Normal{Mu: 0, Sigma: 1}
	zs := mat.NewVecDense(xs.Len(), nil)

	for i := 0; i < xs.Len(); i++ {
		zs.SetVec(i, origF(xs.AtVec(i)+ND.Rand()/5.0))
	}
	return zs
}

func randF(a, b float64) float64 {
	return rand.Float64()*(b-a) + a
}

func getBestModelIndex(weights mat.Vector) int {
	bestIndex := 0
	bestWeight := weights.AtVec(0)
	for i := 1; i < weights.Len(); i++ {
		if weights.AtVec(i) > bestWeight {
			bestIndex = i
			bestWeight = weights.AtVec(i)
		}
	}
	return bestIndex
}

func drawparticleFilerStep(name string, Xs *mat.VecDense, fs *filterState, ts, Z mat.Vector) {
	p := plot.New()
	pts := make(plotter.XYs, Xs.Len())
	var draws []plot.Plotter

	for i := range pts {
		pts[i].X = Xs.AtVec(i)
		pts[i].Y = origF(pts[i].X)
	}
	t, _ := plotter.NewLine(pts)
	t.LineStyle.Width = 2
	t.LineStyle.Color = color.RGBA{R: 255, A: 255}
	p.Legend.Add("Trues", t)
	draws = append(draws, t)

	for lineIndex := 0; lineIndex < fs.θs.RawMatrix().Rows; lineIndex++ {
		for i := range pts {
			pts[i].X = Xs.AtVec(i)
			pts[i].Y = fFmodel(pts[i].X, fs.θs.RowView(lineIndex))
		}
		t, _ := plotter.NewLine(pts)
		t.LineStyle.Width = 1
		// t.LineStyle.Color = color.RGBA{R: 255, A: 255}
		// p.Legend.Add("Trues", t)
		draws = append(draws, t)
	}

	bestIModelIndex := getBestModelIndex(fs.weights)
	for i := range pts {
		pts[i].X = Xs.AtVec(i)
		pts[i].Y = fFmodel(pts[i].X, fs.θs.RowView(bestIModelIndex))
	}
	best, _ := plotter.NewLine(pts)
	best.LineStyle.Width = 2
	best.LineStyle.Color = color.RGBA{G: 255, A: 255}
	p.Legend.Add("Best current model", best)
	draws = append(draws, best)

	pts = make(plotter.XYs, ts.Len())
	for i := range pts {
		pts[i].X = ts.AtVec(i)
		pts[i].Y = Z.AtVec(i)
	}
	lpPoints, _ := plotter.NewScatter(pts)
	lpPoints.Shape = draw.PyramidGlyph{}
	lpPoints.Color = color.RGBA{B: 255, A: 255}
	draws = append(draws, lpPoints)

	// p.HideAxes()
	p.Add(draws...)
	p.Title.Text = "Фильтр частиц"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"
	p.X.Min = 0
	p.X.Max = 2
	p.Y.Min = 1
	p.Y.Max = 7

	if err := p.Save(1000, 800, name+".jpg"); err != nil {
		panic(err)
	}
}
