#include <stdio.h>
#include <time.h>
#include <stdlib.h>

float data[][2] = {
  {0, 0},
  {1, 2},
  {2, 4},
  {3, 6},
  {4, 8},
  {5, 10},
  {6, 12},
};

#define DATA_COUNT (sizeof(data) / sizeof(data[0]))

// y = weight * x
float cost(float weight, float bias)
{
  float result = 0.0f;
  for (size_t i = 0; i < DATA_COUNT; i++)
  {
    float y = data[i][0] * weight + bias;
    float distance = y - data[i][1];
    result += distance*distance;
  }

  return (result);
}
float create_rand()
{
  return (((float)rand() / (float)RAND_MAX));
}

int main()
{
  srand(69);
  float weight = create_rand() * 10.0f;
  float epsilon = 1e-3;
  float bias = create_rand() * 5.0f;
  float learning_rate = 1e-3;
  printf("bias: %f\n", bias);
  printf("weight: %f\n",weight);
  float cost_it = cost(weight, bias);
  int iteration = 0;
  while (cost_it >= 1)
  {
    float dW = (cost(weight + epsilon, bias) - cost(weight, bias)) / epsilon; 
    float dB = (cost(weight, bias + epsilon) - cost(weight, bias)) / epsilon; 
    weight -= learning_rate * dW;
    bias -= learning_rate * dB;
    cost_it = cost(weight, bias);
    iteration++;
    //printf("i:%d ==> cost: %f\n",iteration, cost_it);
  }
  printf("iteration: %d\n", iteration);
  printf("weight: %f\n", weight);
  printf("bias: %f\n", bias);
} 
