// DBN
package GoDeep

import (
	"fmt"
)

type DBN struct {
	RBMLayers []RBM
}

func NewDBN(Network []int) *DBN {
	Result := new(DBN)
	Result.RBMLayers = make([]RBM, len(Network)-1)
	for i := 0; i < len(Network)-1; i++ {
		Result.RBMLayers[i] = *NewRBM(Network[i], Network[i+1])
	}
	return Result
}

func (Self *DBN) Train(TrainData *[][]float64, Config Train) float64 {
	var Activation [][]float64

	Activation = *TrainData
	for i := range Self.RBMLayers {
		Error := Self.RBMLayers[i].Train(&Activation, Config)
		fmt.Printf("Layer %d Error %f\r\n", i, Error)
		for j := range Activation {
			Activation[j] = *Self.RBMLayers[i].Forward(&Activation[j])
		}
	}
	return 0.0
}

func (Self *DBN) Forward(Input *[]float64) *[]float64 {
	var Activation []float64
	Activation = *Input
	for i := range Self.RBMLayers {
		Activation = *Self.RBMLayers[i].Forward(&Activation)
	}
	return &Activation
}

func (Self *DBN) Feedback(Input *[]float64) *[]float64 {
	var Activation []float64
	Activation = *Input
	for i := range Self.RBMLayers {
		Activation = *Self.RBMLayers[len(Self.RBMLayers)-1-i].Feedback(&Activation)
	}
	return &Activation
}

func (Self *DBN) ToNN() *NN {
	Network := make([]int, len(Self.RBMLayers)+1)
	for i := 0; i < len(Self.RBMLayers); i++ {
		Network[i] = len(Self.RBMLayers[i].BiasV)
	}
	Network[len(Self.RBMLayers)] = len(Self.RBMLayers[len(Self.RBMLayers)-1].BiasH)

	Result := new(NN)
	Result.Weight = make([][][]float64, len(Network)-1)
	Result.Bias = make([][]float64, len(Network)-1)
	Result.DeltaWeight = make([][][]float64, len(Network)-1)
	for i := range Result.Weight {
		Result.Weight[i] = NewMatrix(Network[i], Network[i+1])
		Result.Bias[i] = make([]float64, Network[i+1])
		Result.DeltaWeight[i] = NewMatrix(Network[i], Network[i+1])
	}
	for i := range Result.Weight {
		for j := range Result.Weight[i] {
			for k := range Result.Weight[i][j] {
				Result.Weight[i][j][k] = Self.RBMLayers[i].Weight[j][k]
			}
		}
	}
	for i := range Result.Bias {
		for j := range Result.Bias[i] {
			Result.Bias[i][j] = Self.RBMLayers[i].BiasH[j]
		}
	}
	return Result
}
