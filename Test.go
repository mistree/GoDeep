// NNTest
package main

import (
	"fmt"
	"github.com/mistree/GoDeep"
	"runtime"
	"time"
)

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	//NNTest()
	//RBMTest()
	DBNTest()
}

func NNTest() {
	nn := GoDeep.NewNN([]int{2, 10, 10, 1})
	inputs := [][]float64{
		[]float64{0, 0},
		[]float64{0, 1},
		[]float64{1, 0},
		[]float64{1, 1},
	}

	targets := [][]float64{
		[]float64{0},
		[]float64{1},
		[]float64{1},
		[]float64{0},
	}

	var Trainer GoDeep.Train
	Trainer.Dropout = 0
	Trainer.Epoch = 1000
	Trainer.LearnRate = 0.1
	Trainer.Momentum = 0.1
	Trainer.ErrorAllowed = 0.1
	start := time.Now()
	Error, Succeed := nn.Train(&inputs, &targets, Trainer)
	fmt.Printf("NN Finished training in: %s\r\n", time.Since(start))
	if Succeed {
		fmt.Printf("Succeed, Error %f\r\n", Error)
	} else {
		fmt.Printf("Failed, Error %f\r\n", Error)
	}
	fmt.Println(*nn.Forward(&inputs[0]))
	fmt.Println(*nn.Forward(&inputs[1]))
	fmt.Println(*nn.Forward(&inputs[2]))
	fmt.Println(*nn.Forward(&inputs[3]))
}

func RBMTest() {
	m := GoDeep.NewRBM(2, 4)
	var Trainer GoDeep.Train
	Trainer.Dropout = 0.1
	Trainer.LearnRate = 1
	Trainer.Momentum = 0.1
	Trainer.Epoch = 1000
	Trainer.ErrorAllowed = 0.1
	start := time.Now()
	Error := m.Train(&[][]float64{[]float64{0, 0},
		[]float64{0, 1},
		[]float64{1, 0},
		[]float64{1, 1}}, Trainer)
	fmt.Printf("RBM Finished training in: %s\r\n", time.Since(start))
	fmt.Printf("Error %f\r\n", Error)
	fmt.Println(*m.Feedback(m.Forward(&[]float64{0, 0})))
	fmt.Println(*m.Feedback(m.Forward(&[]float64{0, 1})))
	fmt.Println(*m.Feedback(m.Forward(&[]float64{1, 0})))
	fmt.Println(*m.Feedback(m.Forward(&[]float64{1, 1})))
}

func DBNTest() {
	d := GoDeep.NewDBN([]int{2, 5, 5, 1})
	var Trainer GoDeep.Train
	Trainer.Dropout = 0.1
	Trainer.LearnRate = 1
	Trainer.Momentum = 0.1
	Trainer.Epoch = 5000
	Trainer.ErrorAllowed = 0 //.1
	d.Train(&[][]float64{
		[]float64{0, 0},
		[]float64{0, 1},
		[]float64{1, 0},
		[]float64{1, 1}}, Trainer)
	fmt.Println("Reconstruction:")
	fmt.Println(*d.Feedback(d.Forward(&[]float64{0, 0})))
	fmt.Println(*d.Feedback(d.Forward(&[]float64{0, 1})))
	fmt.Println(*d.Feedback(d.Forward(&[]float64{1, 0})))
	fmt.Println(*d.Feedback(d.Forward(&[]float64{1, 1})))

	///*
	n := d.ToNN()
	Trainer.Epoch = 1000
	Trainer.Dropout = 0
	n.Train(&[][]float64{
		[]float64{0, 0},
		[]float64{0, 1},
		[]float64{1, 0},
		[]float64{1, 1}},
		&[][]float64{
			[]float64{0},
			[]float64{1},
			[]float64{1},
			[]float64{0}}, Trainer)
	fmt.Println("Forward:")
	fmt.Println(*n.Forward(&[]float64{0, 0}))
	fmt.Println(*n.Forward(&[]float64{0, 1}))
	fmt.Println(*n.Forward(&[]float64{1, 0}))
	fmt.Println(*n.Forward(&[]float64{1, 1}))
	//*/
}
