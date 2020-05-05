
#include <cmath>
#include <algorithm>
#include <iostream>

#include "matrix.h"
#include "neural.h"

void disp_matrix(const Matrix &matrix) {
  cout<<"disp_matrix(matrix) called with "<<matrix.rows<<" rows and "<<matrix.cols<<" cols: "<<endl;
  for (int r=0;r<matrix.rows;r++)
  {
    cout<<"Row "<<r<<": ";
    for (int c=0;c<matrix.cols;c++)
    {
      cout<<matrix(r,c)<< ", ";
    }
    cout<<endl;
  }
  //cout<<"for matrix, width="<<len<<", height="<<sum;
  cout<<endl;
}

void disp_matrix(const Matrix &matrix, int row) {
  cout<<"disp_matrix(matrix,row) called for row "<<row<<" with "<<matrix.rows<<" rows and "<<matrix.cols<<" len (cols): "<<endl;
  //for (int r=0;r<matrix.rows;r++)
  //{
    double sum = 0;
    cout<<"Row "<<row<<": ";
    for (int c=0;c<matrix.cols;c++)
    {
      cout<<matrix(row,c)<< ", ";
      sum = sum + matrix(row,c);
    }
    cout<<endl;
  //}
  cout<<"for Row "<<row<<", sum ="<<sum;
  cout<<endl;
}

// Calculate a linear activation (i.e. no activation).
//  f(x) = x
// Parameters:
//  Matrix &matrix: the input non-activated output of the layer.
// Returns:
//  A Matrix containing the activated output.
Matrix forward_linear(const Matrix &matrix) {
  Matrix activated = matrix;
  // TODO: Implement forward activation.
  //for(int )
  /*
  for (int r=0;r<matrix.rows;r++)
  {
    for (int c=0;c<matrix.cols;c++)
    {
      activated(r,c) = matrix(r,c);
      //cout<<activated(r,c)<<endl;
    }
  }
  disp_matrix(matrix,7);
  //cout<<activated[0].data<<endl;

  */
  //NOT_IMPLEMENTED();
  return activated;
}

// Calculate the backwards pass for the activation.
// Parameters:
//  const Matrix& out: the activated output of the current layer.
//  const Matrix& prev_grad: the gradient from the next layer (towards the Loss).
// Returns:
//  Matrix: the gradients of this layer (to be passed to the previous layer).
Matrix backward_linear(const Matrix &out, const Matrix &prev_grad) {
  assert_same_size(prev_grad, out);
  Matrix grad = prev_grad;
  // TODO: Implement activation backward pass.
  NOT_IMPLEMENTED();
  return grad;
}

// Calculate a logistic activation (sigmoid).
// Parameters:
//  Matrix &matrix: the input non-activated output of the layer.
// Returns:
//  A Matrix containing the activated output.
Matrix forward_logistic(const Matrix &matrix) {
  Matrix activated = matrix;
  // TODO: Implement forward activation.
  // Hint: look at matrix.h, it might save you some typing.
  for (int r=0;r<matrix.rows;r++)
  {
    for (int c=0;c<matrix.cols;c++)
    {
      activated(r,c) = 1/(1+ exp(-1*matrix(r,c)));
    }
  }
  //NOT_IMPLEMENTED();
  return activated;
}

// Calculate the backwards pass for the activation.
// Parameters:
//  const Matrix& out: the activated output of the current layer.
//  const Matrix& prev_grad: the gradient from the next layer (towards the Loss).
// Returns:
//  Matrix: the gradients of this layer (to be passed to the previous layer).
Matrix backward_logistic(const Matrix &out, const Matrix &prev_grad) {
  assert_same_size(prev_grad, out);
  Matrix grad = prev_grad;
  // TODO: Implement activation backward pass.
  for (int r=0;r<grad.rows;r++)
  {
    for (int c=0;c<grad.cols;c++)
    {
      grad(r,c) = (out(r,c)*(1-out(r,c)))*prev_grad(r,c);
    }
  }
  //NOT_IMPLEMENTED();
  return grad;
}

// Calculate a tanh activation.
// Parameters:
//  Matrix &matrix: the input non-activated output of the layer.
// Returns:
//  A Matrix containing the activated output.
Matrix forward_tanh(const Matrix &matrix) {
  Matrix activated = matrix;
  // TODO: Implement forward activation.
  for (int r=0;r<matrix.rows;r++)
  {
    for (int c=0;c<matrix.cols;c++)
    {
      activated(r,c) = tanh(matrix(r,c));
    }
  }
  //NOT_IMPLEMENTED();
  return activated;
}

// Calculate the backwards pass for the activation.
// Parameters:
//  const Matrix& out: the activated output of the current layer.
//  const Matrix& prev_grad: the gradient from the next layer (towards the Loss).
// Returns:
//  Matrix: the gradients of this layer (to be passed to the previous layer).
Matrix backward_tanh(const Matrix &out, const Matrix &prev_grad) {
  assert_same_size(prev_grad, out);
  Matrix grad = prev_grad;
  // TODO: Implement activation backward pass.
  for (int r=0;r<grad.rows;r++)
  {
    for (int c=0;c<grad.cols;c++)
    {
      grad(r,c) = (1-(out(r,c)*out(r,c)))*prev_grad(r,c);
    }
  }
  //NOT_IMPLEMENTED();
  return grad;
}

// Calculate a ReLU activation.
// Parameters:
//  Matrix &matrix: the input non-activated output of the layer.
// Returns:
//  A Matrix containing the activated output.
Matrix forward_relu(const Matrix &matrix) {
  Matrix activated = matrix;
  // TODO: Implement forward activation.
  for (int r=0;r<matrix.rows;r++)
  {
    for (int c=0;c<matrix.cols;c++)
    {
      if (activated(r,c)<0) { activated(r,c)=0; }
    }
  }
  //NOT_IMPLEMENTED();
  return activated;
}

// Calculate the backwards pass for the activation.
// Parameters:
//  const Matrix& out: the activated output of the current layer.
//  const Matrix& prev_grad: the gradient from the next layer (towards the Loss).
// Returns:
//  Matrix: the gradients of this layer (to be passed to the previous layer).
Matrix backward_relu(const Matrix &out, const Matrix &prev_grad) {
  assert_same_size(prev_grad, out);
  Matrix grad = prev_grad;
  // TODO: Implement activation backward pass.
  for (int r=0;r<grad.rows;r++)
  {
    for (int c=0;c<grad.cols;c++)
    {
      if (out(r,c)<0) { grad(r,c) = 0; }
      else { grad(r,c) = prev_grad(r,c); }
    }
  }
  //NOT_IMPLEMENTED();
  return grad;
}

// Calculate a Leaky ReLU activation.
// Use slope = 0.01
// Parameters:
//  Matrix &matrix: the input non-activated output of the layer.
// Returns:
Matrix forward_lrelu(const Matrix &matrix) {
  Matrix activated = matrix;
  // TODO: Implement forward activation.
  for (int r=0;r<matrix.rows;r++)
  {
    for (int c=0;c<matrix.cols;c++)
    {
      if (activated(r,c)<0) { activated(r,c)=activated(r,c)*.01; }
    }
  }
  NOT_IMPLEMENTED();
  return activated;
}

// Calculate the backwards pass for the activation.
// Parameters:
//  const Matrix& out: the activated output of the current layer.
//  const Matrix& prev_grad: the gradient from the next layer (towards the Loss).
// Returns:
//  Matrix: the gradients of this layer (to be passed to the previous layer).
Matrix backward_lrelu(const Matrix &out, const Matrix &prev_grad) {
  assert_same_size(prev_grad, out);
  Matrix grad = prev_grad;
  // TODO: Implement activation backward pass.
  for (int r=0;r<grad.rows;r++)
  {
    for (int c=0;c<grad.cols;c++)
    {
      grad(r,c) = prev_grad(r,c);
      if (out(r,c)<0) { grad(r,c) = .01 * grad(r,c); }
      //if (out(r,c)<0) { grad(r,c) = .01 * prev_grad(r,c); }
      //else { grad(r,c) = prev_grad(r,c); }
    }
  }
  //NOT_IMPLEMENTED();
  return grad;
}

// Calculate a Softmax activation.
// Parameters:
//  Matrix &matrix: the input non-activated output of the layer.
// Returns:
Matrix forward_softmax(const Matrix &matrix) {
  Matrix activated = matrix;
  // TODO: Implement forward activation.
  for (int r=0;r<matrix.rows;r++)
  {
    double sum = 0;
    
    for (int c=0;c<matrix.cols;c++)
    {
      
      activated(r,c)=exp(activated(r,c)); 
      sum += activated(r,c);
    }
    //cout<<sum<<",";
    for (int c=0;c<matrix.cols;c++)
    {
      activated(r,c)=activated(r,c)/sum; 
    }

  }
  /*
  cout<<endl;
  disp_matrix(matrix,63);
  disp_matrix(activated,63);
  */
  //NOT_IMPLEMENTED();
  return activated;
}

// Computes the Jacobian of the softmax function for a single row.
//
// Parameters:
//  Matrix &out_row: a 1xM vector matrix representing the output activation of a softmax function.
// Returns:
//  an MxM matrix representing the Jacobian matrix.
Matrix softmax_jacobian(const Matrix &out_row) {
  assert(out_row.rows == 1);
  Matrix jacobian(out_row.cols, out_row.cols);
  // TODO: Implement the Jacobian matrix.
  // Do whatever you want here, but here's some structure to get you started.
  for (int j = 0; j < out_row.cols; j++) {
    for (int k = 0; k < out_row.cols; k++) {
      // compute if diagonal
      if (j == k) {
        jacobian(j,k) = out_row(k) - out_row(k) * out_row(k);
      }
      else {
        jacobian(j,k) = -1 * out_row(k) * out_row(j);
      }
      // compute if off diagonal
      //NOT_IMPLEMENTED();
      // jacobian(j, k) = ...
    }
  }
  /*
  cout<<"disp_matrix(out_row) and disp_matrix(jacobian,0):"<<endl;
  disp_matrix(out_row,0);
  disp_matrix(jacobian,0);
  */
  assert(jacobian.rows == out_row.cols);
  assert(jacobian.cols == out_row.cols);
  return jacobian;
}

// Computes the backwards pass for the softmax function.
Matrix backward_softmax(const Matrix &out, const Matrix &prev_grad) {
  assert_same_size(prev_grad, out);
  // TODO: Implement activation backward pass.
  Matrix grad = prev_grad;
  // Multiply previous gradient with Jacobian.
  for (int i = 0; i < out.rows; i++) {
    Matrix jacobian = softmax_jacobian(out.get_row(i));
    Matrix row_grad = prev_grad.get_row(i);
    // TODO: Implement the softmax backward pass.
    //NOT_IMPLEMENTED();
    //Matrix grad_update = jacobian * row_grad;
    Matrix grad_update = row_grad * jacobian;
    // grad(i, j) = ...
    // grad_ = jacobian * row_grad;
    
    for ( int j=0;j<out.cols;j++)
    {
      //grad(i,j) = jacobian(i,j)* prev_grad(i,j); // might be out of bounds on jacobian b/c cols only (j,j instead?)?
      //grad(i,j) = jacobian * row_grad;
      grad(i,j) = grad_update(j);
    }
    
    
    
  }
  //disp_matrix(out,0);
  return grad;
}

// Run an activation function on each element in a matrix,
//
// Matrix& m: Input to activation function
// Activation a: function to run
// return the activated matrix
Matrix forward_activate_matrix(const Matrix &matrix, Activation a) {
  if (a == LINEAR) {
    return forward_linear(matrix);
  } else if (a == LOGISTIC) {
    return forward_logistic(matrix);
  } else if (a == TANH) {
    return forward_tanh(matrix);
  } else if (a == RELU) {
    return forward_relu(matrix);
  } else if (a == LRELU) {
    return forward_lrelu(matrix);
  } else if (a == SOFTMAX) {
    return forward_softmax(matrix);
  } else {
    assert(false); // Invalid activation.
  }
}

// Calculates the gradient of an activation function
// and multiplies it into the initial gradient for a layer
//
// const Matrix& out: an activated layer output
// Activation a: activation function for a layer
// Matrix& grad: before activation gradient (initial layer gradient)
// returns: Matrix that is after applying the activation gradien
Matrix backward_activate_matrix(const Matrix &out, const Matrix &grad, Activation a) {
  if (a == LINEAR) {
    return backward_linear(out, grad);
  } else if (a == LOGISTIC) {
    return backward_logistic(out, grad);
  } else if (a == TANH) {
    return backward_tanh(out, grad);
  } else if (a == RELU) {
    return backward_relu(out, grad);
  } else if (a == LRELU) {
    return backward_lrelu(out, grad);
  } else if (a == SOFTMAX) {
    return backward_softmax(out, grad);
  } else {
    assert(false); // Invalid activation.
  }
}
