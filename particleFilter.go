package main

import (
	"fmt"
	"image/color"
	"io/ioutil"
	"math"
	"math/rand"
	"sync"

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

func shiftθ(θ []float64) []float64 {
	result := make([]float64, len(θ))
	copy(result, θ)
	result[0] += randF(-0.1, 0.1)
	result[1] += randF(-0.1, 0.1)
	result[2] += randF(-0.1, 0.1)
	result[3] += randF(-0.1, 0.1)
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
	var chans = []chan bool{
		make(chan bool),
		make(chan bool),
	}
	a := func(begin, end int, chanIndex int) {
		for i := begin; i < end; i++ {
			deltaValue := calcDelta(t, Z, fs.θs.RowView(i))
			fs.weights.SetVec(i, 1/deltaValue)
		}
		chans[chanIndex] <- true
	}
	go a(0, fs.weights.Len()/2, 0)
	go a(fs.weights.Len()/2, fs.weights.Len(), 1)
	<-chans[0]
	<-chans[1]

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
	uniqueIndeces := make(map[int]bool)
	for i := 0; i < fs.θs.RawMatrix().Rows; i++ {
		betta += UD.Rand()
		for betta > fs.weights.AtVec(index) {
			betta -= fs.weights.AtVec(index)
			index = (index + 1) % fs.weights.Len()
		}

		if uniqueIndeces[index] == false {
			uniqueIndeces[index] = true
			newθs.SetRow(i, fs.θs.RawRowView(index))
			newWeights.SetVec(i, fs.weights.AtVec(index))
		} else {
			newθs.SetRow(i, shiftθ(fs.θs.RawRowView(index)))
			newWeights.SetVec(i, 1.0/float64(fs.weights.Len()))
		}
	}

	fs.θs = newθs
	fs.weights = newWeights
	// запоминаем индексы, если больше одного, то шифтим
}

func launchParticleFilter() string {
	N := 100
	deltaT := float64(2) / float64(N-1)
	X := mat.NewVecDense(N, nil)
	for i := 0; i < X.Len(); i++ {
		X.SetVec(i, float64(i)*deltaT)
	}

	fs := newFilterState(200)
	Z := getNoiseData(X, origF)

	var wg sync.WaitGroup
	wg.Add(N)
	for i := 0; i < N; i++ {
		fs.updateWeights(X.SliceVec(0, i+1), Z.SliceVec(0, i+1))
		// fmt.Printf("weights: %0.4v\n", mat.Formatted(filterState.weights, mat.Prefix("         ")))

		if (i+1)%5 == 0 {
			fs.updateθs()
		}
		// drawparticleFilterStep("pf"+fmt.Sprint(i), X, fs, X.SliceVec(0, i+1), Z.SliceVec(0, i+1))

		copyTheta := fs.θs
		copyTheta.Copy(fs.θs)
		copyWeights := fs.weights
		copyWeights.CopyVec(fs.weights)
		go func(i int) {
			drawparticleFilterStep("pf"+fmt.Sprint(i), X, &filterState{θs: copyTheta, weights: copyWeights}, X.SliceVec(0, i+1), Z.SliceVec(0, i+1))
			wg.Done()
		}(i)
		index := fs.getBestModelIndex()
		as := fs.θs.RawRowView(index)
		fmt.Printf("%0.2f*e^(%0.2fx%+0.2f)+%0.2f\n", as[0], as[1], as[2], as[3])
	}
	wg.Wait()

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
	return ""
}

func getNoiseData(xs mat.Vector, realFunction func(float64) float64) *mat.VecDense {
	ND := distuv.Normal{Mu: 0, Sigma: 1}
	zs := mat.NewVecDense(xs.Len(), nil)

	for i := 0; i < xs.Len(); i++ {
		zs.SetVec(i, origF(xs.AtVec(i)+ND.Rand()/10.0))
	}
	return zs
}

func randF(a, b float64) float64 {
	return rand.Float64()*(b-a) + a
}

func (fs filterState) getBestModelIndex() int {
	bestIndex := 0
	bestWeight := fs.weights.AtVec(0)
	for i := 1; i < fs.weights.Len(); i++ {
		if fs.weights.AtVec(i) > bestWeight {
			bestIndex = i
			bestWeight = fs.weights.AtVec(i)
		}
	}
	return bestIndex
}

func drawparticleFilterStep(name string, Xs *mat.VecDense, fs *filterState, ts, Z mat.Vector) {
	p := plot.New()
	pts := make(plotter.XYs, Xs.Len())
	var draws []plot.Plotter

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

	for i := range pts {
		pts[i].X = Xs.AtVec(i)
		pts[i].Y = origF(pts[i].X)
	}
	t, _ := plotter.NewLine(pts)
	t.LineStyle.Width = 2
	t.LineStyle.Color = color.RGBA{R: 255, A: 255}
	p.Legend.Add("Trues", t)
	draws = append(draws, t)

	bestIModelIndex := fs.getBestModelIndex()
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
	lpPoints.Color = color.RGBA{R: 255, B: 255, A: 255}
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
