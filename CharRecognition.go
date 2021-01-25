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
	DIM            = 196
	DATASIZE       = 200
	TRAIN_DATASIZE = 180
	TEST_DATASIZE  = 20
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
		path := fmt.Sprintf("./data/1/%03d.txt", i+1)
		datasetA[i] = readFile(path)
		path = fmt.Sprintf("./data/2/%03d.txt", i+1)
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
	eigenVec *Matrix // データから求まった固有値最大の固有ベクトル

	attributeA      []float64 // 次元削減後のデータ
	attributeB      []float64 // 次元削減後のデータ
	attributeMeanA  float64   // 固有ベクトルの方向へ射影した後のAの平均値
	attributeMeanB  float64   // 固有ベクトルの方向へ射影した後のBの平均値
	attributeSigmaA float64   // 固有ベクトルの方向へ射影した後のAの標準偏差
	attributeSigmaB float64   // 固有ベクトルの方向へ射影した後のBの標準偏差

}

// constructor
func NewLDA(datasetA [DATASIZE][DIM]float64, datasetB [DATASIZE][DIM]float64, experiment_mode int) *LDA {
	lda := new(LDA)
	trainA := make([][DIM]float64, TRAIN_DATASIZE)
	trainB := make([][DIM]float64, TRAIN_DATASIZE)
	testA := make([][DIM]float64, TEST_DATASIZE)
	testB := make([][DIM]float64, TEST_DATASIZE)

	if experiment_mode == 1 {
		_ = copy(testA, datasetA[0:20])
		_ = copy(testB, datasetB[0:20])
		_ = copy(trainA, datasetA[20:DATASIZE])
		_ = copy(trainB, datasetB[20:DATASIZE])
	} else {
		var testX, testY, trainX, trainY, trainX2, trainY2 int
		switch experiment_mode {
		case 2:
			testX, testY = 20, 40
			trainX, trainY = 0, 20
			trainX2, trainY2 = 40, DATASIZE
		case 3:
			testX, testY = 40, 60
			trainX, trainY = 0, 40
			trainX2, trainY2 = 60, DATASIZE
		case 4:
			testX, testY = 60, 80
			trainX, trainY = 0, 60
			trainX2, trainY2 = 80, DATASIZE
		case 5:
			testX, testY = 80, 100
			trainX, trainY = 0, 80
			trainX2, trainY2 = 100, DATASIZE
		}
		// おそらくlda.test~にポインタごと入ってしまっているので，appendしたときにもとの配列のポインタ（形状）が変わってしまっている？
		_ = copy(testA, datasetA[testX:testY])
		_ = copy(testB, datasetB[testX:testY])
		_ = copy(trainA, append(datasetA[trainX:trainY], datasetA[trainX2:trainY2]...))
		_ = copy(trainB, append(datasetB[trainX:trainY], datasetB[trainX2:trainY2]...))
	}
	lda.trainA = trainA
	lda.trainB = trainB
	lda.testA = testA
	lda.testB = testB
	return lda
}

/* 各カテゴリの平均・全データの平均を求める
 * @return aveA, aveB, AveAll
 */
func (l *LDA) average() (*Matrix, *Matrix, *Matrix) {
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
	aveA = totalA.times(1 / float64(TRAIN_DATASIZE))
	aveB = totalB.times(1 / float64(TRAIN_DATASIZE))
	// 辞書全体の平均
	aveAll = (aveA.add(aveB)).times(1.0 / 2.0)
	return aveA, aveB, aveAll
}

//　　カテゴリ間分散Sb・カテゴリ内分散Swの最大・最小化（のための）項を求める
func (l *LDA) variance() (*Matrix, *Matrix) {
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

func (l *LDA) train() {
	const MODE = 1
	// fmt.Printf("--- sb/sw ---\n")
	sb, sw := l.variance()
	// fmt.Printf("--- eigenVec/val ---\n")
	_, eigenvec := sw.inverse().product(sb).eigenMax1()
	l.eigenVec = eigenvec
	// fmt.Println(l.eigenVec)
	// 射影
	// fmt.Println(l.trainA)
	l.attributeA = make([]float64, len(l.trainA))
	l.attributeB = make([]float64, len(l.trainB))
	for idx, tA := range l.trainA {
		tA := NewMatrixByVector(tA[:])
		l.attributeA[idx] = tA.T().product(l.eigenVec).mat[0][0]
		tB := NewMatrixByVector(l.trainB[idx][:])
		l.attributeB[idx] = tB.T().product(l.eigenVec).mat[0][0]
	}

	sumA := 0.0
	sumB := 0.0

	for idx, aA := range l.attributeA {
		sumA += aA
		sumB += l.attributeB[idx]
	}
	l.attributeMeanA = sumA / float64(len(l.trainA))
	l.attributeMeanB = sumB / float64(len(l.trainB))

	sigmaA := 0.0
	sigmaB := 0.0

	for idx, aA := range l.attributeA {
		sigmaA += math.Pow(aA-l.attributeMeanA, 2)
		sigmaB += math.Pow(l.attributeB[idx]-l.attributeMeanB, 2)
	}
	l.attributeSigmaA = math.Sqrt(sigmaA / float64(len(l.trainA)))
	l.attributeSigmaB = math.Sqrt(sigmaB / float64(len(l.trainB)))
}

func (l *LDA) trainResult() {
	// fmt.Println(l.attributeA)
	// fmt.Println(l.attributeB)
	fmt.Println(l.attributeMeanA)
	fmt.Println(l.attributeMeanB)
	fmt.Println(l.attributeSigmaA)
	fmt.Println(l.attributeSigmaB)
}

// 予測 A --- 1, B -- -1
func (l *LDA) predict(input [DIM]float64) int {
	i := NewMatrixByVector(input[:])
	x := i.T().product(l.eigenVec).mat[0][0]
	dA := math.Abs(x-l.attributeMeanA) / l.attributeSigmaA
	dB := math.Abs(x-l.attributeMeanB) / l.attributeSigmaB
	if dA < dB {
		return 1
	}
	return -1
}

func (l *LDA) test() (float64, float64) {
	trueA := 0.0
	trueB := 0.0
	for idx, testA := range l.testA {
		if l.predict(testA) == 1 {
			trueA += 1.0
		}
		if l.predict(l.testB[idx]) == -1 {
			trueB += 1.0
		}
	}
	fmt.Println("[字種1] 認識率(%):", trueA/float64(len(l.testA))*100, "正解数/対象データ数:", strconv.Itoa(int(trueA))+"/"+strconv.Itoa(len(l.testA)))
	fmt.Println("[字種2] 認識率(%):", trueB/float64(len(l.testB))*100, "正解数/対象データ数:", strconv.Itoa(int(trueB))+"/"+strconv.Itoa(len(l.testB)))
	fmt.Println("[全体] 認識率(%):", (trueA+trueB)/float64(len(l.testB)*2)*100,
		"正解数/対象データ数: ", strconv.Itoa(int(trueA+trueB))+"/"+strconv.Itoa(2*len(l.testA)))

	return trueA, trueB
}

func main() {
	datasetA, datasetB := readDataset()
	mode := []int{1, 2, 3, 4, 5}
	trueA := 0.0
	trueB := 0.0
	for _, md := range mode {
		fmt.Println("実験番号:", md)
		lda := NewLDA(datasetA, datasetB, md)
		lda.train()
		tA, tB := lda.test()
		trueA += tA
		trueB += tB
		fmt.Println()
	}
	fmt.Println("---[実験全体の認識率]---")
	fmt.Println("[字種2] 認識率(%): ", trueA, "正解数/対象データ数:", strconv.Itoa(int(trueA))+"/100")
	fmt.Println("[字種2] 認識率(%): ", trueB, "正解数/対象データ数:", strconv.Itoa(int(trueB))+"/100")
	fmt.Println("[合計] 認識率(%): ", ((trueA+trueB)/200.0)*100, "正解数/対象データ数:", strconv.Itoa(int(trueA+trueB))+"/200")
}

/**

実験結果:

実験番号: 1
[字種1] 認識率(%): 95 正解数/対象データ数: 19/20
[字種2] 認識率(%): 80 正解数/対象データ数: 16/20
[全体] 認識率(%): 87.5 正解数/対象データ数:  35/40

実験番号: 2
[字種1] 認識率(%): 70 正解数/対象データ数: 14/20
[字種2] 認識率(%): 80 正解数/対象データ数: 16/20
[全体] 認識率(%): 75 正解数/対象データ数:  30/40

実験番号: 3
[字種1] 認識率(%): 80 正解数/対象データ数: 16/20
[字種2] 認識率(%): 70 正解数/対象データ数: 14/20
[全体] 認識率(%): 75 正解数/対象データ数:  30/40

実験番号: 4
[字種1] 認識率(%): 85 正解数/対象データ数: 17/20
[字種2] 認識率(%): 70 正解数/対象データ数: 14/20
[全体] 認識率(%): 77.5 正解数/対象データ数:  31/40

実験番号: 5
[字種1] 認識率(%): 55.00000000000001 正解数/対象データ数: 11/20
[字種2] 認識率(%): 65 正解数/対象データ数: 13/20
[全体] 認識率(%): 60 正解数/対象データ数:  24/40

---[実験全体の認識率]---
[字種1] 認識率(%):  77 正解数/対象データ数: 77/100
[字種2] 認識率(%):  73 正解数/対象データ数: 73/100
[合計] 認識率(%):  75 正解数/対象データ数: 150/200

*/
