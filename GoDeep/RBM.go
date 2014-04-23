// RBM
package GoDeep

import (
	"math"
	"math/rand"
)

type RBM struct {
	Weight      [][]float64
	DeltaWeight [][]float64
	BiasH       []float64
	DeltaBiasH  []float64
	BiasV       []float64
	DeltaBiasV  []float64
}

type Train struct {
	Epoch        int
	LearnRate    float64
	Momentum     float64
	Dropout      float64
	ErrorAllowed float64
}

type _RBMBatch struct {
	DeltaWeight [][]float64
	DeltaBiasH  []float64
	DeltaBiasV  []float64
}

func NewRBM(Visible int, Hidden int) *RBM {
	m := new(RBM)
	m.Weight = NewMatrix(Visible, Hidden)
	m.DeltaWeight = NewMatrix(Visible, Hidden)
	m.BiasH = make([]float64, Hidden)
	m.DeltaBiasH = make([]float64, Hidden)
	m.BiasV = make([]float64, Visible)
	m.DeltaBiasV = make([]float64, Visible)

	for i := range m.Weight {
		for j := range m.Weight[i] {
			m.Weight[i][j] = float64(1.0 / math.Sqrt(float64(Hidden+Visible)))
		}
	}
	return m
}

func (Self *RBM) Train(Visible *[][]float64, Config Train) float64 {
	DeltaWeight := NewMatrix(len(Self.BiasV), len(Self.BiasH))
	DeltaBiasH := make([]float64, len(Self.BiasH))
	DeltaBiasV := make([]float64, len(Self.BiasV))

	BatchNumber := len(*Visible)
	Batch := make(chan *_RBMBatch, BatchNumber)
	Finish := make(chan bool, 3)
	for q := 0; q < Config.Epoch; q++ {
		RandomTable := rand.Perm(BatchNumber)
		for i := 0; i < BatchNumber; i++ {
			go func(Index int) {
				Batch <- Self._TrainBatch(&(*Visible)[RandomTable[Index]], Config)
			}(i)
		}

		for i := 0; i < BatchNumber; i++ {
			Delta := <-Batch

			go func() {
				for j := range Delta.DeltaWeight {
					for k := range Delta.DeltaWeight[j] {
						DeltaWeight[j][k] += Delta.DeltaWeight[j][k]
					}
				}
				Finish <- true
			}()
			go func() {
				for j := range Delta.DeltaBiasH {
					DeltaBiasH[j] += Delta.DeltaBiasH[j]
				}
				Finish <- true
			}()
			go func() {
				for j := range Delta.DeltaBiasV {
					DeltaBiasV[j] += Delta.DeltaBiasV[j]
				}
				Finish <- true
			}()
			<-Finish
			<-Finish
			<-Finish
		}

		go func() {
			for j := range DeltaWeight {
				for k := range DeltaWeight[j] {
					DeltaWeight[j][k] /= float64(len(*Visible))
				}
			}
			for i := range Self.Weight {
				for j := range Self.Weight[i] {
					Self.Weight[i][j] += DeltaWeight[i][j]
				}
			}
			Finish <- true
		}()
		go func() {
			for j := range DeltaBiasH {
				DeltaBiasH[j] /= float64(len(*Visible))
			}
			for i := range Self.BiasH {
				Self.BiasH[i] += DeltaBiasH[i]
			}
			Finish <- true
		}()
		go func() {
			for j := range DeltaBiasV {
				DeltaBiasV[j] /= float64(len(*Visible))
			}
			for i := range Self.BiasV {
				Self.BiasV[i] += DeltaBiasV[i]
			}
			Finish <- true
		}()

		<-Finish
		<-Finish
		<-Finish

		Error := Self.Error(Visible)
		if Error < Config.ErrorAllowed {
			return Error
		}
	}

	return Self.Error(Visible)
}

func (Self *RBM) _TrainBatch(Visible *[]float64, Config Train) *_RBMBatch {
	var Hidden []float64
	var StepVisible []float64
	var StepHidden []float64

	DeltaWeight := NewMatrix(len(Self.BiasV), len(Self.BiasH))
	DeltaBiasH := make([]float64, len(Self.BiasH))
	DeltaBiasV := make([]float64, len(Self.BiasV))

	Correlation1 := NewMatrix(len(Self.BiasV), len(Self.BiasH))
	Correlation2 := NewMatrix(len(Self.BiasV), len(Self.BiasH))

	if Config.Dropout == 0 {
		Hidden = *_Rand(Self.Forward(Visible))
		StepVisible = *Self.Feedback(&Hidden)
		StepHidden = *_Rand(Self.Forward(&StepVisible))
	} else {
		Hidden = *Self._Drop(Self.Forward(Visible), Config.Dropout)
		StepVisible = *Self.Feedback(&Hidden)
		StepHidden = *Self._Drop(Self.Forward(&StepVisible), Config.Dropout)
	}

	for i := range *Visible {
		for j := range Hidden {
			Correlation1[i][j] = (*Visible)[i] * Hidden[j]
		}
	}

	for i := range StepVisible {
		for j := range StepHidden {
			Correlation2[i][j] = StepVisible[i] * StepHidden[j]
		}
	}

	Finish := make(chan bool, 3)

	go func() {
		for i := range Self.DeltaWeight {
			for j := range Self.DeltaWeight[i] {
				DeltaWeight[i][j] = Config.Momentum*Self.DeltaWeight[i][j] + Config.LearnRate*(Correlation1[i][j]-Correlation2[i][j])
			}
		}
		Finish <- true
	}()
	go func() {
		for i := range Self.DeltaBiasH {
			DeltaBiasH[i] = Config.Momentum*Self.DeltaBiasH[i] + Config.LearnRate*(Hidden[i]-StepHidden[i])
		}
		Finish <- true
	}()
	go func() {
		for i := range Self.DeltaBiasV {
			DeltaBiasV[i] = Config.Momentum*Self.DeltaBiasV[i] + Config.LearnRate*((*Visible)[i]-StepVisible[i])
		}
		Finish <- true
	}()

	<-Finish
	<-Finish
	<-Finish

	return &_RBMBatch{DeltaWeight, DeltaBiasH, DeltaBiasV}
}

func (Self *RBM) Forward(Input *[]float64) *[]float64 {
	Result := make([]float64, len(Self.BiasH))
	for i := range Result {
		for j := range *Input {
			Result[i] += Self.Weight[j][i] * (*Input)[j]
		}
		Result[i] = 1 / (1 + math.Exp(-Result[i]-Self.BiasH[i]))
	}
	return &Result
}

func (Self *RBM) Feedback(Input *[]float64) *[]float64 {
	Result := make([]float64, len(Self.BiasV))
	for i := range Result {
		for j := range *Input {
			Result[i] += Self.Weight[i][j] * (*Input)[j]
		}
		Result[i] = 1 / (1 + math.Exp(-Result[i]-Self.BiasV[i]))
	}
	return &Result
}

func (Self *RBM) Error(Visible *[][]float64) float64 {
	var Error float64
	for i := range *Visible {
		Hidden := Self.Forward(&(*Visible)[i])
		StepVisible := Self.Feedback(Hidden)
		for j := range *StepVisible {
			Error += math.Abs((*Visible)[i][j] - (*StepVisible)[j])
		}
	}
	Error /= float64(len(*Visible))
	return Error
}

func (Self *RBM) _Drop(Input *[]float64, Rate float64) *[]float64 {
	for i := range *Input {
		if rand.Float64() < Rate {
			(*Input)[i] = 0
		}
	}
	return Input
}

func _Rand(Input *[]float64) *[]float64 {
	for i := range *Input {
		if (*Input)[i] > rand.Float64() {
			(*Input)[i] = 1
		} else {
			(*Input)[i] = 0
		}
	}
	return Input
}
