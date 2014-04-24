// DBN
package GoDeep

import (
	"math/rand"
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
	var Error float64
	Activation = *TrainData
	for i := 0; i < len(Self.RBMLayers)-1; i++ { // except last layer
		Error += Self.RBMLayers[i].Train(&Activation, Config)
		for j := range Activation {
			Activation[j] = *Self.RBMLayers[i].RoundForward(&Activation[j])
		}
	}
	return Error / float64(len(Self.RBMLayers)-1)
}

func (Self *DBN) Forward(Input *[]float64) *[]float64 {
	var Activation []float64
	Activation = *Input
	for i := range Self.RBMLayers {
		Activation = *Self.RBMLayers[i].Forward(&Activation)
	}
	return &Activation
}

func AppendLabel(Activation, Label *[][]float64) *[][]float64 {
	Result := make([][]float64, len(*Activation))
	LabelNumber := len(*Label)
	RandomTable := rand.Perm(LabelNumber)
	for i := range Result {
		Result[i] = append((*Activation)[i], (*Label)[RandomTable[i%LabelNumber]]...)
	}
	return &Result
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

	// init last layer randomly
	for j := range Result.Weight[len(Result.Weight)-1] {
		for k := range Result.Weight[len(Result.Weight)-1][j] {
			Result.Weight[len(Result.Weight)-1][j][k] = rand.Float64()*2 - 1
			//Result.Weight[len(Result.Weight)-1][j][k] = (rand.Float64() - 0.5) * Bound * 2
		}
	}

	for j := range Result.Bias[len(Result.Weight)-1] {
		Result.Bias[len(Result.Weight)-1][j] = 0
	}

	return Result
}
