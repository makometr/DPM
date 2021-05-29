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

type model struct {
	θs     *mat.VecDense
	weight float64
}

func initModels(N int) []model {
	models := make([]model, N)
	cfsNumber := 4

	for i := 0; i < N; i++ {
		models[i] = model{
			weight: 1.0 / float64(N),
			θs: mat.NewVecDense(cfsNumber, []float64{
				randF(0.2, 1.1),
				randF(0.2, 1.1),
				randF(0.2, 1.1),
				randF(0.2, 1.1),
			})}
	}
	return models
}

func getNoiseData(xs mat.Vector, realFunction func(float64) float64) *mat.VecDense {
	ND := distuv.Normal{Mu: 0, Sigma: 1}
	zs := mat.NewVecDense(xs.Len(), nil)

	for i := 0; i < xs.Len(); i++ {
		zs.SetVec(i, origF(xs.AtVec(i)+ND.Rand()/5.0))
	}
	return zs
}

func fFmodel(x float64, θ mat.Vector) float64 {
	c0, c1, c2, c3 := θ.AtVec(0), θ.AtVec(1), θ.AtVec(2), θ.AtVec(3)
	return c0*math.Exp(c1*x+c2) + c3
}

func initParticles(N int, lowBorder, highBorder float64) (*mat.Dense, *mat.VecDense) {
	θs := make([]float64, N*4)
	for i := 0; i < N*4; i++ {
		θs[i] = randF(lowBorder, highBorder)
	}

	weights := make([]float64, N)
	for i := 0; i < N; i++ {
		weights[i] = 1.0 / float64(N)
	}

	return mat.NewDense(N, 4, θs), mat.NewVecDense(N, weights)
}

func observe(Z, t mat.Vector, θs *mat.Dense, weights *mat.VecDense) *mat.VecDense {
	for i := 0; i < θs.RawMatrix().Rows; i++ {
		deltaValue := delta(t, Z, θs.RowView(i))
		if deltaValue < 0.0001 {
			weights.SetVec(i, 1000000.0)
		} else {
			weights.SetVec(i, 1/deltaValue)
		}
	}
	normalizeWeights(weights)
	return weights
}

func resample(θs *mat.Dense, weights *mat.VecDense) (*mat.Dense, *mat.VecDense) {
	newθs := mat.NewDense(θs.RawMatrix().Rows, θs.RawMatrix().Cols, nil)
	newWeights := mat.NewVecDense(weights.Len(), nil)

	index := rand.Intn(θs.RawMatrix().Rows)
	betta := 0.0

	UD := distuv.Uniform{}
	UD.Min = 0
	UD.Max = mat.Max(weights) * 2
	for i := 0; i < θs.RawMatrix().Rows; i++ {
		betta += UD.Rand()
		for betta > weights.AtVec(index) {
			betta -= weights.AtVec(index)
			index = (index + 1) % 20
		}
		newθs.SetRow(i, θs.RawRowView(index))
		newWeights.SetVec(i, weights.AtVec(index))
	}

	// запоминаем индексы, если больше одного, то шифтим
	return newθs, newWeights
}

func doFilter() {
	N := 20
	deltaT := float64(2) / (20 - 1)
	X := mat.NewVecDense(20, nil)
	for i := 0; i < X.Len(); i++ {
		X.SetVec(i, float64(i)*deltaT)
	}

	// Z := mat.NewVecDense(N, nil)
	// t := mat.NewVecDense(N, nil)

	θs, weights := initParticles(N, 0.2, 1.1)
	Z := getNoiseData(X, origF)
	// ND := distuv.Normal{Mu: 0, Sigma: 1}

	for i := 0; i < N; i++ {
		weights = observe(Z.SliceVec(0, i+1), X.SliceVec(0, i+1), θs, weights)

		fmt.Printf("weights: %0.4v\n", mat.Formatted(θs, mat.Prefix("         ")))

		if (i+1)%4 == 0 {
			θs, weights = resample(θs, weights)
		}
		drawparticleFilerStep("pf"+fmt.Sprint(i), X, weights, θs, X.SliceVec(0, i+1), Z.SliceVec(0, i+1))
	}

	checkErr := func(err error) {
		if err != nil {
			panic(err)
		}
	}

	aw, err := mjpeg.New("test.avi", 1000, 800, 2)
	checkErr(err)

	for i := 0; i < N; i++ {
		data, err := ioutil.ReadFile(fmt.Sprintf("pf"+"%d.jpg", i))
		checkErr(err)
		checkErr(aw.AddFrame(data))
	}

	checkErr(aw.Close())
}

func applyExponent(xs *mat.VecDense) *mat.VecDense {
	result := mat.NewVecDense(xs.Len(), nil)
	for i := 0; i < xs.Len(); i++ {
		result.SetVec(i, math.Exp(xs.AtVec(i)))
	}
	return result
}

func origF(elem float64) float64 {
	return math.Exp(elem)
}

func randF(a, b float64) float64 {
	return rand.Float64()*(b-a) + a
}

func delta(t, Z mat.Vector, θ mat.Vector) (result float64) {
	for i := 0; i < Z.Len(); i++ {
		result += math.Pow(math.Abs(Z.AtVec(i)-fFmodel(t.AtVec(i), θ)), 2)
	}
	return result
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

func normalizeWeights(weights *mat.VecDense) {
	sum := mat.Sum(weights)
	for i := 0; i < weights.Len(); i++ {
		weights.SetVec(i, weights.AtVec(i)/sum)
	}
}

func drawparticleFilerStep(name string, Xs, weights *mat.VecDense, thetas *mat.Dense, ts, Z mat.Vector) {
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

	for lineIndex := 0; lineIndex < thetas.RawMatrix().Rows; lineIndex++ {
		for i := range pts {
			pts[i].X = Xs.AtVec(i)
			pts[i].Y = fFmodel(pts[i].X, thetas.RowView(lineIndex))
		}
		t, _ := plotter.NewLine(pts)
		t.LineStyle.Width = 1
		// t.LineStyle.Color = color.RGBA{R: 255, A: 255}
		// p.Legend.Add("Trues", t)
		draws = append(draws, t)
	}

	bestIModelIndex := getBestModelIndex(weights)
	for i := range pts {
		pts[i].X = Xs.AtVec(i)
		pts[i].Y = fFmodel(pts[i].X, thetas.RowView(bestIModelIndex))
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
