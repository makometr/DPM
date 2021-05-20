package main

import "gonum.org/v1/gonum/mat"

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
