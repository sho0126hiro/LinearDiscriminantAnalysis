package main

import (
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"strconv"
	"strings"
)

const (
	DIM      = 2
	DATASIZE = 3
)

func stringToFloat64(s []string) [DIM]float64 {
	var r [DIM]float64
	for i, v := range s {
		r[i], _ = strconv.ParseFloat(v, 64)
	}
	return r
}

func readFile(path string) [DIM]float64 {
	b, err := ioutil.ReadFile(path)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	lines := strings.Split(string(b), "\r\n") // 改行コード: CRLF
	return stringToFloat64(lines[:len(lines)-1])
}

func readDataset() ([DATASIZE][DIM]float64, [DATASIZE][DIM]float64) {
	var datasetA [DATASIZE][DIM]float64
	var datasetB [DATASIZE][DIM]float64
	for i := 0; i < DATASIZE; i++ {
		path := fmt.Sprintf("./testdata/1/%03d.txt", i+1)
		datasetA[i] = readFile(path)
		path = fmt.Sprintf("./testdata/2/%03d.txt", i+1)
		datasetB[i] = readFile(path)
	}
	return datasetA, datasetB
}

type Matrix struct {
	mat [][]float64
	row int
	col int
}

func NewMatrix(m [][]float64) *Matrix {
	matrix := new(Matrix)
	matrix.col = len(m[0])
	matrix.row = len(m)
	matrix.mat = m
	return matrix
}

func NewZeroMatrix(row, col int) *Matrix {
	matrix := new(Matrix)
	matrix.col = col
	matrix.row = row
	m := make([][]float64, row)
	for i := 0; i < row; i++ {
		m[i] = make([]float64, col)
		for j := 0; j < col; j++ {
			m[i][j] = 0.0
		}
	}
	matrix.mat = m
	return matrix
}

func NewMatrixByVector(m []float64) *Matrix {
	base := make([][]float64, 1)
	base[0] = m
	return NewMatrix(base).T()
}

func (m Matrix) identity() *Matrix {
	result := NewZeroMatrix(m.row, m.row)
	for i := 0; i < m.row; i++ {
		result.mat[i][i] = 1.0
	}
	return result
}

// m * target を実装（m*target: 行列の掛け算）
func (m Matrix) product(target *Matrix) *Matrix {
	result := NewZeroMatrix(m.row, target.col)
	for k := 0; k < m.row; k++ {
		for j := 0; j < target.col; j++ {
			tmp := 0.0
			for i := 0; i < target.row; i++ {
				tmp += m.mat[k][i] * target.mat[i][j]
			}
			result.mat[k][j] = tmp
		}
	}
	return result
}

// 行列全要素定数倍
func (m Matrix) times(k float64) *Matrix {
	result := NewZeroMatrix(m.row, m.col)
	for i := 0; i < m.row; i++ {
		for j := 0; j < m.col; j++ {
			result.mat[i][j] = k * m.mat[i][j]
		}
	}
	return result
}

// 同じ行・列の行列を加算する(m + target)
func (m Matrix) add(target *Matrix) *Matrix {
	result := NewZeroMatrix(m.row, m.col)
	for i := 0; i < m.row; i++ {
		for j := 0; j < m.col; j++ {
			result.mat[i][j] = m.mat[i][j] + target.mat[i][j]
		}
	}
	return result
}

// 行列の引き算
func (m Matrix) sub(target *Matrix) *Matrix {
	result := NewZeroMatrix(m.row, m.col)
	for j := 0; j < m.row; j++ {
		for i := 0; i < m.col; i++ {
			result.mat[j][i] = m.mat[j][i] - target.mat[j][i]
		}
	}
	return result
}

func (m Matrix) T() *Matrix {
	result := NewZeroMatrix(m.col, m.row)
	for j := 0; j < m.row; j++ {
		for i := 0; i < m.col; i++ {
			result.mat[i][j] = m.mat[j][i]
		}
	}
	return result
}

// 全く同じMatrixを新たに生成する（値渡し）
func (m Matrix) copy() *Matrix {
	result := NewZeroMatrix(m.row, m.col)
	for j := 0; j < m.row; j++ {
		for i := 0; i < m.col; i++ {
			result.mat[j][i] = m.mat[j][i]
		}
	}
	return result
}

// m(matrix)の逆行列を求める(行列は正方行列)
// Gauss-Jordan Method
// https://www.cs.tsukuba.ac.jp/~oyama/isyspro2019/ex/ex1.html
func (m Matrix) inverse() *Matrix {
	result := NewZeroMatrix(m.row, m.col).identity()
	mCopy := m.copy()
	for j := 0; j < mCopy.row; j++ {
		pivot := mCopy.mat[j][j]
		for i := 0; i < mCopy.col; i++ {
			mCopy.mat[j][i] /= pivot
			result.mat[j][i] /= pivot
		}
		for i := 0; i < mCopy.col; i++ {
			if i != j {
				tmp := mCopy.mat[i][j]
				for k := 0; k < mCopy.col; k++ {
					mCopy.mat[i][k] -= mCopy.mat[j][k] * tmp
					result.mat[i][k] -= result.mat[j][k] * tmp
				}
			}
		}
	}
	return result
}

func (m Matrix) toSlice() [][]float64 {
	return m.mat
}

// 行列の中の絶対値最大のインデックスを取得する
// 行, 列
func (m Matrix) getAbsMaxIndex() (int, int) {
	tmp := 0.0
	rowMaxIndex, colMaxIndex := 0, 0
	for j := 0; j < m.row; j++ {
		for i := j + 1; i < m.col; i++ {
			if math.Abs(m.mat[j][i]) > tmp {
				tmp = math.Abs(m.mat[j][i])
				rowMaxIndex = j
				colMaxIndex = i
			}
		}
	}
	return rowMaxIndex, colMaxIndex
}

// 回転角の取得
func (m Matrix) getTheta(i, j int) float64 {
	tmp := m.mat[j][j] - m.mat[i][i]
	if tmp == 0 {
		return math.Pi / 4.0
	}
	return 0.5 * math.Atan(2.0*m.mat[i][j]/tmp)
}

func (m Matrix) eigenMax1() (float64, *Matrix) {
	// 最大固有値・最大固有ベクトルのみを求めて返す
	const (
		KMAX = 1000    // 最大繰り返し回数
		EPS  = 0.00001 // 収束判定条件
	)
	v := make([]float64, DIM)
	for i := 0; i < DIM; i++ {
		if i == 0 {
			v[i] = 1.0
		} else {
			v[i] = 0.0
		}
	}

	copyVec := func(v []float64) []float64 {
		result := make([]float64, DIM)
		for i := 0; i < DIM; i++ {
			result[i] = v[i]
		}
		return result
	}
	getProduct := func(a [][]float64, x []float64) []float64 {
		var sum float64
		tmp := make([]float64, DIM)
		for i := 0; i < DIM; i++ {
			tmp[i] = 0.0
		}
		for i := 0; i < DIM; i++ {
			sum = 0.0
			for j := 0; j < DIM; j++ {
				sum += x[j] * a[i][j]
			}
			tmp[i] = sum
		}
		return tmp
	}
	getEpsilon := func(v1, v2 []float64) float64 {
		sum := 0.0
		for i := 0; i < DIM; i++ {
			sum += (v2[i] - v1[i]) * (v2[i] - v1[i])
		}
		return math.Sqrt(sum)
	}
	innerPrduct := func(v1, v2 []float64) float64 {
		sum := 0.0
		for i := 0; i < DIM; i++ {
			sum += v1[i] * v2[i]
		}
		return sum
	}
	vecTimes := func(v []float64, c float64) []float64 {
		result := make([]float64, DIM)
		for i := 0; i < DIM; i++ {
			result[i] = v[i] * c
		}
		return result
	}
	norm := func(v []float64) float64 {
		sum := 0.0
		for i := 0; i < DIM; i++ {
			sum += v[i] * v[i]
		}
		return math.Sqrt(sum)
	}

	var eigenValue float64
	k := 1
	for {
		vTmp := copyVec(v)
		v = getProduct(m.toSlice(), v)
		eigenValue = innerPrduct(v, vTmp)
		v = vecTimes(v, 1/norm(v))
		if getEpsilon(v, vTmp) < EPS {
			break
		}
		if k > KMAX {
			break
		}
		k += 1
	}
	return eigenValue, NewMatrixByVector(v)
}

type LDA struct {
	trainA   [][DIM]float64
	trainB   [][DIM]float64
	testA    [][DIM]float64
	testB    [][DIM]float64
	eigenVec *Matrix
}

// constructor
func NewLDA(datasetA [DATASIZE][DIM]float64, datasetB [DATASIZE][DIM]float64, experiment_mode int) *LDA {
	lda := new(LDA)
	// lda.testA = datasetA[0:20]
	// lda.testB = datasetB[0:20]
	// lda.trainA = datasetA[20:DATASIZE]
	// lda.trainB = datasetB[20:DATASIZE]
	lda.trainA = datasetA[:]
	lda.trainB = datasetB[:]
	return lda
}

/* 各カテゴリの平均・全データの平均を求める
 * @return aveA, aveB, AveAll
 */
func (l LDA) average() (*Matrix, *Matrix, *Matrix) {
	aveA := NewMatrixByVector(make([]float64, DIM))
	aveB := NewMatrixByVector(make([]float64, DIM))
	aveAll := NewMatrixByVector(make([]float64, DIM))
	totalA := NewMatrixByVector(make([]float64, DIM))
	totalB := NewMatrixByVector(make([]float64, DIM))
	// 各辞書ごとの平均
	for j, _ := range l.trainA {
		trainA := NewMatrixByVector(l.trainA[j][:])
		trainB := NewMatrixByVector(l.trainB[j][:])
		totalA = totalA.add(trainA)
		totalB = totalB.add(trainB)
	}
	aveA = totalA.times(1 / float64(DATASIZE))
	aveB = totalB.times(1 / float64(DATASIZE))
	// 辞書全体の平均
	aveAll = (aveA.add(aveB)).times(1.0 / 2.0)
	return aveA, aveB, aveAll
}

//　　カテゴリ間分散Sb・カテゴリ内分散Swの最大・最小化（のための）項を求める
func (l LDA) variance() (*Matrix, *Matrix) {
	// クラス内分散最大化項
	sw := NewZeroMatrix(DIM, DIM)
	aveA, aveB, _ := l.average()
	for i, _ := range l.trainA {
		trainA := NewMatrixByVector(l.trainA[i][:])
		trainB := NewMatrixByVector(l.trainB[i][:])
		tmp := trainA.sub(aveA)
		tmp2 := trainB.sub(aveB)
		tmp3 := tmp.product(tmp.T()).add(tmp2.product(tmp2.T()))
		sw = sw.add(tmp3)
	}
	// クラス間分散最小化項
	tmp := aveA.sub(aveB)
	sb := tmp.product(tmp.T())
	return sb, sw
}

func (l LDA) train() {
	const MODE = 1
	fmt.Printf("--- sb/sw ---\n")
	sb, sw := l.variance()
	fmt.Printf("--- eigenVec/val ---\n")
	_, eigenvec := sw.inverse().product(sb).eigenMax1()
	l.eigenVec = eigenvec
	fmt.Println(l.eigenVec)
}

func main() {
	const MODE = 1
	datasetA, datasetB := readDataset()
	lda := NewLDA(datasetA, datasetB, MODE)
	lda.train()
}
