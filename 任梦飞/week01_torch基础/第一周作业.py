{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 2, 3])"
      ]
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import numpy as np\n",
    "a = [1,2,3]\n",
    "b = np.array(a)\n",
    "b"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 2, 3])"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a1 = np.array([1,2,3])\n",
    "a1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1, 2, 3],\n",
       "       [4, 5, 6],\n",
       "       [7, 8, 9]])"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import numpy as np\n",
    "a2 = np.array([(1,2,3),(4,5,6),(7,8,9)])\n",
    "a2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0., 0., 0.],\n",
       "       [0., 0., 0.]])"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import numpy as np\n",
    "a3 = np.zeros([2,3])\n",
    "a3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 3, 5, 7, 9])"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import numpy as np\n",
    "a4 = np.arange(1,10,2)\n",
    "a4"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(8, 8)"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import numpy as np\n",
    "a5 = np.eye(8)\n",
    "a5 = np.shape(a5)\n",
    "a5"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[[0.81876182, 0.17644272],\n",
       "        [0.06291198, 0.2531723 ],\n",
       "        [0.0439684 , 0.10737073]],\n",
       "\n",
       "       [[0.80816386, 0.27280326],\n",
       "        [0.28790572, 0.36574709],\n",
       "        [0.23506476, 0.94948215]],\n",
       "\n",
       "       [[0.23467275, 0.59758373],\n",
       "        [0.15568949, 0.41986023],\n",
       "        [0.16425442, 0.02951852]]])"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import numpy as np\n",
    "a6 = np.random.random([3,3])\n",
    "a7 = np.random.random([3,3,2])\n",
    "a7\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([-1.00691266,  0.67902223,  1.93217727, -0.71494011,  0.52260309,\n",
       "       -2.78786521, -1.71630284])"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import numpy as np\n",
    "mu,sigma = 0,1\n",
    "np.random.normal(mu,sigma,7)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "np.int64(6)"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import numpy as np\n",
    "a8 = np.array([[[1,2,3],[4,5,6],[7,8,9]]])\n",
    "a8 = a8[0,1,2]\n",
    "a8\n",
    "# for i,j,k in a8:\n",
    "#     print(i+j+k)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(1, 3, 3)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([[[[ 2],\n",
       "         [ 3],\n",
       "         [ 4]],\n",
       "\n",
       "        [[ 5],\n",
       "         [ 7],\n",
       "         [ 6]],\n",
       "\n",
       "        [[ 9],\n",
       "         [12],\n",
       "         [14]]]])"
      ]
     },
     "execution_count": 77,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import numpy as np\n",
    "a9 = np.array([[[2,3,4],[5,7,6],[9,12,14]]])\n",
    "# print(a9.ndim)\n",
    "print(a9.shape)\n",
    "# print(a9.size)\n",
    "# print(a9.dtype)\n",
    "# print(a9.reshape(3,1,3))\n",
    "# print(a9)\n",
    "# print(a9.transpose(1,0,2))\n",
    "# print(a9.T)\n",
    "# print(a9.flatten().shape)\n",
    "a9 = a9[:,:,:,np.newaxis]\n",
    "a9"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0 1 2]\n",
      " [3 4 5]\n",
      " [6 7 8]]\n",
      "[[[0 1 2]]\n",
      "\n",
      " [[3 4 5]]\n",
      "\n",
      " [[6 7 8]]]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\DELL\\AppData\\Local\\Temp\\ipykernel_20640\\1383023701.py:6: RuntimeWarning: invalid value encountered in divide\n",
      "  c= a10[:,np.newaxis,:]/b1\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([[[nan,  1.,  1.]],\n",
       "\n",
       "       [[ 1.,  1.,  1.]],\n",
       "\n",
       "       [[ 1.,  1.,  1.]]])"
      ]
     },
     "execution_count": 88,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import numpy as np\n",
    "a10 = np.arange(9).reshape(3,3)\n",
    "b1 = np.arange(9).reshape(3,1,3)\n",
    "print(a10)\n",
    "print(b1)\n",
    "c= a10[:,np.newaxis,:]/b1\n",
    "c"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[[1 2 3]\n",
      "  [4 5 6]\n",
      "  [7 8 9]]]\n",
      "45\n",
      "362880\n",
      "5.0\n",
      "6.666666666666667\n",
      "2.581988897471611\n",
      "9\n",
      "1\n",
      "8\n",
      "0\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "a11 = np.arange(1,10).reshape(1,3,3)\n",
    "print(a11)\n",
    "print(a11.sum())\n",
    "print(a11.prod())\n",
    "print(a11.mean())\n",
    "print(a11.var())\n",
    "print(a11.std())\n",
    "print(a11.max())\n",
    "print(a11.min())\n",
    "print(a11.argmax())\n",
    "print(a11.argmin())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[ 2. -2.  3.  7. -4.]\n",
      "[ 1. -3.  3.  6. -5.]\n",
      "[ 1. -2.  3.  7. -5.]\n",
      "[-4.8 -2.1  1.3  3.   6.9]\n",
      "[ 6.9  3.   1.3 -2.1 -4.8]\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "a12 = np.array([1.3,-2.1,3.0,6.9,-4.8])\n",
    "print(np.ceil(a12))\n",
    "print(np.floor(a12))\n",
    "print(np.rint(a12))\n",
    "print(np.sort(a12))\n",
    "print(np.sort(a12)[::-1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "tensor([[1, 2],\n",
      "        [3, 4]])\n",
      "tensor([[1, 1],\n",
      "        [1, 1]])\n",
      "tensor([[0.8884, 0.4602],\n",
      "        [0.1371, 0.7322]], dtype=torch.float64)\n"
     ]
    }
   ],
   "source": [
    "import torch\n",
    "import numpy as np\n",
    "data = [[1,2],[3,4]]\n",
    "# x_data = torch.tensor(data)\n",
    "np_array = np.array(data)\n",
    "x_data = torch.from_numpy(np_array)\n",
    "print(x_data)\n",
    "x_ones = torch.ones_like(x_data)\n",
    "print(x_ones)\n",
    "x_rand = torch.rand_like(x_data,dtype=float)\n",
    "print(x_rand)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "pyrmf",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
