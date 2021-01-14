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
		path := fmt.Sprintf("../testdata/1/%03d.txt", i+1)
		datasetA[i] = readFile(path)
		path = fmt.Sprintf("../testdata/2/%03d.txt", i+1)
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

func (ele Matrix) identity() *Matrix {
	result := NewZeroMatrix(ele.row, ele.row)
	for i := 0; i < ele.row; i++ {
		result.mat[i][i] = 1.0
	}
	return result
}

// ele * target を実装（ele*target: 行列の掛け算）
func (ele Matrix) product(target *Matrix) *Matrix {
	result := NewZeroMatrix(ele.row, target.col)
	for k := 0; k < ele.row; k++ {
		for j := 0; j < target.col; j++ {
			tmp := 0.0
			for i := 0; i < target.row; i++ {
				tmp += ele.mat[k][i] * target.mat[i][j]
			}
			result.mat[k][j] = tmp
		}
	}
	return result
}

// 行列全要素定数倍
func (ele Matrix) times(k float64) *Matrix {
	result := NewZeroMatrix(ele.row, ele.col)
	for i := 0; i < ele.row; i++ {
		for j := 0; j < ele.col; j++ {
			result.mat[i][j] = k * ele.mat[i][j]
		}
	}
	return result
}

// 同じ行・列の行列を加算する(ele + target)
func (ele Matrix) add(target *Matrix) *Matrix {
	result := NewZeroMatrix(ele.row, ele.col)
	for i := 0; i < ele.row; i++ {
		for j := 0; j < ele.col; j++ {
			result.mat[i][j] = ele.mat[i][j] + target.mat[i][j]
		}
	}
	return result
}

// 行列の引き算
func (ele Matrix) sub(target *Matrix) *Matrix {
	result := NewZeroMatrix(ele.row, ele.col)
	for j := 0; j < ele.row; j++ {
		for i := 0; i < ele.col; i++ {
			result.mat[j][i] = ele.mat[j][i] - target.mat[j][i]
		}
	}
	return result
}

func (ele Matrix) T() *Matrix {
	result := NewZeroMatrix(ele.col, ele.row)
	for j := 0; j < ele.row; j++ {
		for i := 0; i < ele.col; i++ {
			result.mat[i][j] = ele.mat[j][i]
		}
	}
	return result
}

// 全く同じMatrixを新たに生成する（値渡し）
func (ele Matrix) copy() *Matrix {
	result := NewZeroMatrix(ele.row, ele.col)
	for j := 0; j < ele.row; j++ {
		for i := 0; i < ele.col; i++ {
			result.mat[i][j] = ele.mat[i][j]
		}
	}
	return result
}

// ele(matrix)の逆行列を求める(行列は正方行列)
// Gauss-Jordan Method
// https://www.cs.tsukuba.ac.jp/~oyama/isyspro2019/ex/ex1.html
func (ele Matrix) inverse() *Matrix {
	result := NewZeroMatrix(ele.row, ele.col).identity()
	eleCopy := ele.copy()
	for j := 0; j < eleCopy.row; j++ {
		pivot := eleCopy.mat[j][j]
		for i := 0; i < eleCopy.col; i++ {
			eleCopy.mat[j][i] /= pivot
			result.mat[j][i] /= pivot
		}
		for i := 0; i < eleCopy.col; i++ {
			if i != j {
				tmp := eleCopy.mat[i][j]
				for k := 0; k < eleCopy.col; k++ {
					eleCopy.mat[i][k] -= eleCopy.mat[j][k] * tmp
					result.mat[i][k] -= result.mat[j][k] * tmp
				}
			}
		}
	}
	return result
}

// 行列の中の絶対値最大のインデックスを取得する
// 行, 列
func (ele Matrix) getAbsMaxIndex() (int, int) {
	tmp := 0.0
	rowMaxIndex, colMaxIndex := 0, 0
	for j := 0; j < ele.row; j++ {
		for i := j + 1; i < ele.col; i++ {
			if math.Abs(ele.mat[j][i]) > tmp {
				tmp = math.Abs(ele.mat[j][i])
				rowMaxIndex = j
				colMaxIndex = i
			}
		}
	}
	return rowMaxIndex, colMaxIndex
}

// 回転角の取得
func (ele Matrix) getTheta(i, j int) float64 {
	tmp := ele.mat[j][j] - ele.mat[i][i]
	if tmp == 0 {
		return math.Pi / 4.0
	}
	return 0.5 * math.Atan(2.0*ele.mat[i][j]/tmp)
}

// 固有値と固有ベクトルを返す(Jacobi method)
// return [固有値, 固有ベクトル]
// func (ele Matrix) eigen() ([]float64, *Matrix) {
// 	fmt.Println(ele)
// 	const (
// 		K_MAX = 100000
// 		A_MIN = 0.01
// 	)
// 	E := NewZeroMatrix(DIM, DIM).identity()
// 	P := NewZeroMatrix(DIM, DIM).identity()
// 	for k := 0; k < K_MAX; k++ {
// 		i, j := ele.getAbsMaxIndex()
// 		if math.Abs(ele.mat[i][j]) < A_MIN {
// 			break
// 		}
// 		tmp := ele.copy()
// 		theta := ele.getTheta(i, j)
// 		co := math.Cos(theta)
// 		si := math.Sin(theta)
// 		co2 := math.Cos(theta * 2)
// 		si2 := math.Sin(theta * 2)
// 		for x := 0; x < DIM; x++ {
// 			ele.mat[i][x] = tmp.mat[i][x]*co - tmp.mat[j][x]*si
// 			ele.mat[j][x] = tmp.mat[i][x]*si + tmp.mat[j][x]*co
// 			ele.mat[x][i] = ele.mat[i][x]
// 			ele.mat[x][j] = ele.mat[j][x]
// 		}

// 		ele.mat[i][i] = (tmp.mat[i][i]+tmp.mat[j][j])*0.5 + (tmp.mat[i][i]-tmp.mat[j][j])*co2*0.5 - tmp.mat[i][j]*si2
// 		ele.mat[j][j] = (tmp.mat[i][i]+tmp.mat[j][j])*0.5 - (tmp.mat[i][i]-tmp.mat[j][j])*co2*0.5 + tmp.mat[i][j]*si2
// 		ele.mat[i][j] = 0.0
// 		ele.mat[j][i] = 0.0
// 		now := E.copy()
// 		tmp = P.copy()
// 		now.mat[i][i] = co
// 		now.mat[i][j] = si
// 		now.mat[j][i] = -1.0 * si
// 		now.mat[j][j] = co
// 		for x := 0; x < DIM; x++ {
// 			P.mat[x][i] = tmp.mat[x][i]*now.mat[i][i] + tmp.mat[x][j]*now.mat[j][i]
// 			P.mat[x][j] = tmp.mat[x][i]*now.mat[i][j] + tmp.mat[x][j]*now.mat[j][j]
// 		}
// 	}
// 	eigenvector := P.copy()
// 	fmt.Println(eigenvector)
// 	fmt.Println("EIGENVALUE")
// 	var eigenvalue = make([]float64, DIM)
// 	for i := 0; i < DIM; i++ {
// 		eigenvalue[i] = ele.mat[i][i]
// 	}
// 	fmt.Println(eigenvalue)
// 	return eigenvalue, eigenvector
// }

// PowerMethod
func power(i int) int {
	idx := i
	tmp := 0.0
	for x := 0; x < DIM; x++ {
		tmp += 
	}
	const (
		M    = 1000
		KMAX = 1000
		EPS  = 0.01
	)
	return i
}

// https://www.sist.ac.jp/~suganuma/programming/9-sho/num/pow/pow.htm#pow_cpp
func (ele Matrix) eigen() ([]float64, *Matrix) {
	v0 := NewMatrixByVector(make([]float64, DIM))
	idx := power(0)
}

type LDA struct {
	trainA [][DIM]float64
	trainB [][DIM]float64
	testA  [][DIM]float64
	testB  [][DIM]float64
}

// constructor
func NewLDA(
	datasetA [DATASIZE][DIM]float64, datasetB [DATASIZE][DIM]float64,
	experiment_mode int, // 実験番号
) *LDA {
	lda := new(LDA)
	// lda.testA = datasetA[0:20]
	// lda.testB = datasetB[0:20]
	// lda.trainA = datasetA[20:DATASIZE]
	// lda.trainB = datasetB[20:DATASIZE]
	lda.trainA = datasetA[:]
	lda.trainB = datasetB[:]
	return lda
}

// 各カテゴリの平均・全データの平均を求める
// return aveA, aveB, AveAll 全てMatrix型
func (ele LDA) average() (*Matrix, *Matrix, *Matrix) {
	aveA := NewMatrixByVector(make([]float64, DIM))
	aveB := NewMatrixByVector(make([]float64, DIM))
	aveAll := NewMatrixByVector(make([]float64, DIM))
	totalA := NewMatrixByVector(make([]float64, DIM))
	totalB := NewMatrixByVector(make([]float64, DIM))
	// 各辞書ごとの平均
	for j, _ := range ele.trainA {
		trainA := NewMatrixByVector(ele.trainA[j][:])
		trainB := NewMatrixByVector(ele.trainB[j][:])
		totalA = totalA.add(trainA)
		totalB = totalB.add(trainB)
	}
	aveA = totalA.times(1 / float64(DATASIZE))
	aveB = totalB.times(1 / float64(DATASIZE))
	// 辞書全体の平均
	aveAll = (aveA.add(aveB)).times(1.0 / 2.0)
	return aveA, aveB, aveAll
}

// http://blog.livedoor.jp/itukano/archives/51798055.html
//　　カテゴリ間分散Sb・カテゴリ内分散Swの最大・最小化（のための）項を求める
func (ele LDA) variance() (*Matrix, *Matrix) {
	// クラス内分散最大化項
	sw := NewZeroMatrix(DIM, DIM)
	aveA, aveB, _ := ele.average()
	for i, _ := range ele.trainA {
		trainA := NewMatrixByVector(ele.trainA[i][:])
		trainB := NewMatrixByVector(ele.trainB[i][:])
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

func main() {
	const MODE = 1
	datasetA, datasetB := readDataset()
	lda := NewLDA(datasetA, datasetB, MODE)
	fmt.Printf("--- sb/sw ---\n")
	sb, sw := lda.variance()
	fmt.Printf("--- eigen vec/val ---\n")
	eigenvec, eigenval := sw.inverse().product(sb).eigen()
	fmt.Println(eigenvec)
	fmt.Println(eigenval)
}

/**
１．d次元のデータセットを標準化する(dは特徴量の個数)。
２．クラス毎にd次元の平均ベクトル(各次元の平均値で構成されるベクトル)を計算する。
３．平均ベクトルを使って、クラス間変動行列: クラス間共分散行列とクラス内変動行列: 総クラス内共分散行列を生成する。
４．行列の固有ベクトルと対応する固有値を計算する。
５．固有値を降順でソートすることで、対応する固有ベクトルをランク付けする。
６．d×k次元の変換行列Wを生成するために、最も大きいk個の固有値に対応するk個の固有ベクトルを選択する(固有ベクトルから変換行列Wを生成)。固有ベクトルは、この行列の列である。
７．変換行列Wを使って使ってサンプルを新しい特徴部空間へ射影する。
https://haruchun.jp/lda/
*/
