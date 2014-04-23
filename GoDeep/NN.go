// NN
package GoDeep

import (
	"math"
	"math/rand"
	"time"
)

type NN struct {
	Weight      [][][]float64
	Bias        [][]float64
	DeltaWeight [][][]float64
}

type _NNBatch struct {
	DeltaWeight [][][]float64
	DeltaBias   [][]float64
}

func NewNN(Network []int) *NN {
	Result := new(NN)
	Result.Weight = make([][][]float64, len(Network)-1)
	Result.Bias = make([][]float64, len(Network)-1)
	Result.DeltaWeight = make([][][]float64, len(Network)-1)
	for i := range Result.Weight {
		Result.Weight[i] = NewMatrix(Network[i], Network[i+1])
		Result.Bias[i] = make([]float64, Network[i+1])
		Result.DeltaWeight[i] = NewMatrix(Network[i], Network[i+1])
	}
	rand.Seed(time.Now().UTC().UnixNano())
	for i := range Result.Weight {
		for j := range Result.Weight[i] {
			for k := range Result.Weight[i][j] {
				if i != len(Result.Weight)-1 {
					Result.Weight[i][j][k] = (rand.Float64() - 0.5) * 8 * math.Sqrt(6.0/(float64(len(Result.Weight[i])+len(Result.Weight[i+1]))))
				} else {
					Result.Weight[i][j][k] = rand.Float64()*2 - 1
				}
			}
		}
	}
	return Result
}

func (Self *NN) Train(TrainData *[][]float64, LableData *[][]float64, Config Train) (float64, bool) {
	DeltaWeight := make([][][]float64, len(Self.Weight))
	DeltaBias := make([][]float64, len(Self.Weight))

	for i := 0; i < len(Self.Weight)-1; i++ {
		DeltaBias[i] = make([]float64, len(Self.Weight[i+1]))
		DeltaWeight[i] = NewMatrix(len(Self.Weight[i]), len(Self.Weight[i+1]))
	}
	DeltaBias[len(Self.Weight)-1] = make([]float64, len(Self.Weight[len(Self.Weight)-1][0]))
	DeltaWeight[len(Self.Weight)-1] = NewMatrix(len(Self.Weight[len(Self.Weight)-1]), len(Self.Weight[len(Self.Weight)-1][0]))

	BatchNumber := len(*TrainData)
	Batch := make(chan *_NNBatch, BatchNumber)

	for q := 0; q < Config.Epoch; q++ {
		Finish := make(chan bool, 2)
		// 1-4 go parallel
		RandomTable := rand.Perm(BatchNumber)
		for i := 0; i < BatchNumber; i++ {
			go func(Index int) {
				Batch <- Self._TrainBatch(TrainData, LableData, Config, RandomTable[Index])
			}(i)
		}

		for i := 0; i < BatchNumber; i++ {
			Delta := <-Batch

			go func() {
				for j := range Delta.DeltaWeight {
					for k := range Delta.DeltaWeight[j] {
						for l := range Delta.DeltaWeight[j][k] {
							DeltaWeight[j][k][l] += Delta.DeltaWeight[j][k][l]
						}
					}
				}
				Finish <- true
			}()

			go func() {
				for j := range Delta.DeltaBias {
					for k := range Delta.DeltaBias[j] {
						DeltaBias[j][k] += Delta.DeltaBias[j][k]
					}
				}
				Finish <- true
			}()
			<-Finish
			<-Finish
		}

		// 5 apply delta
		go func() {
			for k := range DeltaWeight {
				for l := range DeltaWeight[k] {
					for n := range DeltaWeight[k][l] {
						Self.Weight[k][l][n] -= (Self.DeltaWeight[k][l][n]*Config.Momentum + DeltaWeight[k][l][n]/float64(len(*TrainData))) * Config.LearnRate
					}
				}
			}
			Finish <- true
		}()

		go func() {
			for k := range Self.Bias {
				for l := range Self.Bias[k] {
					Self.Bias[k][l] -= DeltaBias[k][l] * Config.LearnRate / float64(len(*TrainData))
				}
			}
			Finish <- true
		}()

		<-Finish
		<-Finish
		Self.DeltaWeight = DeltaWeight

		Error := Self.Error(TrainData, LableData)
		if Error <= Config.ErrorAllowed {
			return Error, true
		}
	}
	Error := Self.Error(TrainData, LableData)
	return Error, false
}

func (Self *NN) _TrainBatch(TrainData *[][]float64, LableData *[][]float64, Config Train, Index int) *_NNBatch {
	Activation := make([][]float64, len(Self.Weight)+1)
	Error := make([][]float64, len(Self.Weight)+1)
	DeltaWeight := make([][][]float64, len(Self.Weight))
	DeltaBias := make([][]float64, len(Self.Weight))

	for i := range DeltaWeight {
		Activation[i] = make([]float64, len(Self.Weight[i]))
		Error[i] = make([]float64, len(Self.Weight[i]))
	}
	Activation[len(Self.Weight)] = make([]float64, len(Self.Weight[len(Self.Weight)-1][0]))
	Error[len(Self.Weight)] = make([]float64, len(Self.Weight[len(Self.Weight)-1][0]))

	for i := 0; i < len(Self.Weight)-1; i++ {
		DeltaBias[i] = make([]float64, len(Self.Weight[i+1]))
		DeltaWeight[i] = NewMatrix(len(Self.Weight[i]), len(Self.Weight[i+1]))
	}
	DeltaBias[len(Self.Weight)-1] = make([]float64, len(Self.Weight[len(Self.Weight)-1][0]))
	DeltaWeight[len(Self.Weight)-1] = NewMatrix(len(Self.Weight[len(Self.Weight)-1]), len(Self.Weight[len(Self.Weight)-1][0]))

	// 1 cal activation with bias for each layer
	Activation[0] = (*TrainData)[Index]
	for k := range Self.Weight {
		for l := range Activation[k+1] {
			for n := range Activation[k] {
				Activation[k+1][l] += Activation[k][n] * Self.Weight[k][n][l]
			}
			if rand.Float64() < Config.Dropout {
				Activation[k+1][l] = 0
			} else {
				Activation[k+1][l] = 1 / (1 + math.Exp(-Activation[k+1][l]-Self.Bias[k][l]))
			}
		}
	}

	// 2 cal output Error
	for k := len(Error[len(Error)-1]) - 1; k >= 0; k-- {
		Error[len(Error)-1][k] = -Activation[len(Error)-1][k] * (1.0 - Activation[len(Error)-1][k]) * ((*LableData)[Index][k] - Activation[len(Error)-1][k])
	}

	// 3 cal error feedback
	for k := len(Error) - 2; k >= 1; k-- {
		for l := range Error[k] {
			for n := range Error[k+1] {
				Error[k][l] += Self.Weight[k][l][n] * Error[k+1][n]
			}
			Error[k][l] = Error[k][l] * Activation[k][l] * (1.0 - Activation[k][l])
		}
	}

	// 4 cal delta weight and bias
	for k := range DeltaWeight {
		for l := range Activation[k] {
			for n := range Error[k+1] {
				DeltaWeight[k][l][n] = Error[k+1][n] * Activation[k][l]
			}
		}
	}
	for k := range Self.Bias {
		for l := range Self.Bias[k] {
			DeltaBias[k][l] = Error[k+1][l]
		}
	}

	return &_NNBatch{DeltaWeight, DeltaBias}
}

func (Self *NN) Error(TrainData *[][]float64, LableData *[][]float64) float64 {
	var Error float64
	for i := range *TrainData {
		Result := Self.Forward(&(*TrainData)[i])
		for j := range *Result {
			Error += math.Abs((*Result)[j] - (*LableData)[i][j])
		}
	}
	Error /= float64(len(*TrainData))
	return Error
}

func (Self *NN) Forward(InputData *[]float64) *[]float64 {
	Activation := make([][]float64, len(Self.Weight)+1)
	for i := range Self.Weight {
		Activation[i] = make([]float64, len(Self.Weight[i]))
	}
	Activation[len(Self.Weight)] = make([]float64, len(Self.Weight[len(Self.Weight)-1][0]))

	Activation[0] = *InputData
	for k := range Self.Weight {
		for l := range Activation[k+1] {
			for n := range Activation[k] {
				Activation[k+1][l] += Activation[k][n] * Self.Weight[k][n][l]
			}
			Activation[k+1][l] = 1 / (1 + math.Exp(-Activation[k+1][l]-Self.Bias[k][l]))
		}
	}
	return &Activation[len(Self.Weight)]
}
