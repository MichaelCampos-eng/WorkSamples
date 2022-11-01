#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/*Takes the dot product of two vectors*/
double dot(matrix *mat1, matrix *mat2, int start1, int start2) {
    __m256d sum = _mm256_set1_pd(0.0);
    double total[4];
    int cols = (mat1->cols);

    //Might be something wrong in this for loop
    for(int i = 0; i < cols/16 * 16; i+=16) {
        __m256d m1 = _mm256_loadu_pd(&(mat1->data[start1 + i]));  
        __m256d m2 = _mm256_loadu_pd(&(mat2->data[start2 + i]));
        __m256d mul = _mm256_mul_pd(m1, m2);
        sum = _mm256_add_pd(sum, mul);

        m1 = _mm256_loadu_pd(&(mat1->data[start1 + i+4]));  
        m2 = _mm256_loadu_pd(&(mat2->data[start2 + i+4]));
        mul = _mm256_mul_pd(m1, m2);
        sum = _mm256_add_pd(sum, mul);

        m1 = _mm256_loadu_pd(&(mat1->data[start1 + i+8]));  
        m2 = _mm256_loadu_pd(&(mat2->data[start2 + i+8]));
        mul = _mm256_mul_pd(m1, m2);
        sum = _mm256_add_pd(sum, mul);

        m1 = _mm256_loadu_pd(&(mat1->data[start1 + i+12]));  
        m2 = _mm256_loadu_pd(&(mat2->data[start2 + i+12]));
        mul = _mm256_mul_pd(m1, m2);
        sum = _mm256_add_pd(sum, mul);
    }

    for(int i = cols/16 * 16; i < cols/4 * 4; i+=4) {
        __m256d m1 = _mm256_loadu_pd(&(mat1->data[start1 + i]));  
        __m256d m2 = _mm256_loadu_pd(&(mat2->data[start2 + i]));
        __m256d mul = _mm256_mul_pd(m1, m2);
        sum = _mm256_add_pd(sum, mul);

    }

    _mm256_storeu_pd((__m256d *) total, sum);
    for(int i = cols/4 * 4; i < cols; i++) {
        double prod = (mat1->data[start1 + i]) * (mat2->data[start2 + i]);
        total[0] += prod;
    }
    double finalSum = total[0] + total[1] + total[2] + total[3];
    return finalSum;
}
/*
double dot(matrix *mat1, matrix *mat2, int start1, int start2) {
    __m256d globalSum = _mm256_set1_pd(0.0);
    double total[4];
    double x[1];
    x[0] = 0;
    int cols = (mat1->cols);
    #pragma omp parallel
    {   
        __m256d localSum = _mm256_set1_pd(0.0);
        #pragma omp for
        for(int i = 0; i < cols/4 * 4; i+=4) {
            __m256d m1 = _mm256_loadu_pd(&(mat1->data[start1 + i]));  
            __m256d m2 = _mm256_loadu_pd(&(mat2->data[start2 + i]));
            __m256d mul = _mm256_mul_pd(m1, m2);
            localSum = _mm256_add_pd(localSum, mul);
        }
        #pragma omp critical
        globalSum += localSum; 
    }
    for(int i = cols/4 * 4; i < cols; i++) {
            double prod = (mat1->data[start1 + i]) * (mat2->data[start2 + i]);
            x[0] += prod;
        }
    _mm256_storeu_pd((__m256d *) total, globalSum);
    double finalSum = x[0] + total[0] + total[1] + total[2] + total[3];
    return finalSum;
} */

/*Creates a tranposed matrix*/
matrix *transpose(matrix *mat2){
    int rows = mat2->rows;
    int cols = mat2->cols;

    matrix *matInit;
    allocate_matrix(&matInit, cols, rows);

    #pragma omp parallel 
    {
        #pragma omp for 
        for(int i = 0; i < rows; i += 1){
            for(int j = 0; j < cols; j += 1){
                double val = get(mat2, i, j);
                set(matInit, j, i, val);
            }
        }
    }
    return matInit;
}

/*
void printTrans(matrix *matInit) {
    printf("\n");
    printf("Tranpose Matrix \n");
    for (int y = 0; y < matInit->rows; y++) {
        printf("Row %d: ", y); 
        for (int x = 0; x < matInit->cols; x++) {
            printf("%lf ", get(matInit, y, x));
        }
        printf("\n"); 
    }
}
*/

void printInputMat(matrix *mat2) {
    printf("\n");
    printf("Input Matrix \n");
    for (int y = 0; y < mat2->rows; y++) {
        printf("Row %d: ", y); 
        for (int x = 0; x < mat2->cols; x++) {
            printf("\n");
            printf("%lf ", get(mat2, y, x));
        }
    }
}

/* Generates a random double between low and high */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Returns the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid. Note that the matrix is in row-major order.
 */
double get(matrix *mat, int row, int col) {
    // Task 1.1 TODO
    int j = mat->cols;
    int num = row*j+col;
    return mat->data[num];
}

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid. Note that the matrix is in row-major order.
 */
void set(matrix *mat, int row, int col, double val) {
    // Task 1.1 TODO
    int j = mat->cols;
    int num = row*j+col;
    mat->data[num] = val;

}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. `parent` should be set to NULL to indicate that
 * this matrix is not a slice. You should also set `ref_cnt` to 1.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Return 0 upon success.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    // Task 1.2 TODO
    // HINTS: Follow these steps.
    // 1. Check if the dimensions are valid. Return -1 if either dimension is not positive.
    // 2. Allocate space for the new matrix struct. Return -2 if allocating memory failed.
    // 3. Allocate space for the matrix data, initializing all entries to be 0. Return -2 if allocating memory failed.
    // 4. Set the number of rows and columns in the matrix struct according to the arguments provided.
    // 5. Set the `parent` field to NULL, since this matrix was not created from a slice.
    // 6. Set the `ref_cnt` field to 1.
    // 7. Store the address of the allocated matrix struct at the location `mat` is pointing at.
    // 8. Return 0 upon success.
    if(rows<=0||cols<=0){
        return -1;
    }
    matrix *matter = (matrix*)malloc(sizeof(matrix));
    if(matter==NULL){
        return -2;
    }
    matter->data = (double*)calloc((rows+1)*(cols+1),sizeof(double));

    if(matter->data == NULL){
        return -2;
    }
    
    matter->ref_cnt = 1;
    matter->parent = NULL;
    matter->rows = rows;
    matter->cols = cols;
    *mat = matter;
    return 0;

}

/*
 * You need to make sure that you only free `mat->data` if `mat` is not a slice and has no existing slices,
 * or that you free `mat->parent->data` if `mat` is the last existing slice of its parent matrix and its parent
 * matrix has no other references (including itself).
 */
void deallocate_matrix(matrix *mat) {
    // Task 1.3 TODO
    // HINTS: Follow these steps.
    // 1. If the matrix pointer `mat` is NULL, return.
    // 2. If `mat` has no parent: decrement its `ref_cnt` field by 1. If the `ref_cnt` field becomes 0, then free `mat` and its `data` field.
    // 3. Otherwise, recursively call `deallocate_matrix` on `mat`'s parent, then free `mat`.
    
    if(mat == NULL){
        return;
    }
    if(mat->parent==NULL){
        mat->ref_cnt -= 1;
        if(mat->ref_cnt==0){
            
            free(mat->data);
            free(mat);
        }
    }
    else{
        deallocate_matrix(mat->parent);
        free(mat);
    }

}

/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * Its data should point to the `offset`th entry of `from`'s data (you do not need to allocate memory)
 * for the data field. `parent` should be set to `from` to indicate this matrix is a slice of `from`
 * and the reference counter for `from` should be incremented. Lastly, do not forget to set the
 * matrix's row and column values as well.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Return 0 upon success.
 * NOTE: Here we're allocating a matrix struct that refers to already allocated data, so
 * there is no need to allocate space for matrix data.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) {
    // Task 1.4 TODO
    // HINTS: Follow these steps.
    // 1. Check if the dimensions are valid. Return -1 if either dimension is not positive.
    // 2. Allocate space for the new matrix struct. Return -2 if allocating memory failed.
    // 3. Set the `data` field of the new struct to be the `data` field of the `from` struct plus `offset`.
    // 4. Set the number of rows and columns in the new struct according to the arguments provided.
    // 5. Set the `parent` field of the new struct to the `from` struct pointer.
    // 6. Increment the `ref_cnt` field of the `from` struct by 1.
    // 7. Store the address of the allocated matrix struct at the location `mat` is pointing at.
    // 8. Return 0 upon success.
    if(rows<=0||cols<=0){
        return -1;
    }
    matrix *matter = malloc(sizeof(matrix));
    if(matter==NULL){
        return -2;
    }
    double *da = from->data;
    matter->data = da+offset;

    if(matter->data == NULL){
        return -2;
    }
    
    from->ref_cnt += 1;
    
    matter->parent = from;
    matter->rows = rows;
    matter->cols = cols;
    *mat = matter;
    return 0;
}

/*
 * Sets all entries in mat to val. Note that the matrix is in row-major order.
 */

void fill_matrix(matrix *mat, double val) {
    int num = (mat->rows)*(mat->cols);
    #pragma omp parallel
    {
        #pragma omp for
        for(int x = 0; x < num/4 * 4; x += 4) {
            __m256d vector = _mm256_set1_pd(val);
            _mm256_storeu_pd((__m256d *) &(mat->data[x]), vector);
        }

        for (int x = num/4 * 4; x < num; x++) {
            mat->data[x] = val;
        } 
    }
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success.
 * Note that the matrix is in row-major order.
 */
int abs_matrix(matrix *result, matrix *mat) {
    int num = (mat->rows)*(mat->cols);
    __m256d negOnes = _mm256_set1_pd(-1);

    #pragma omp parallel if (num > 1000)
    {
        #pragma omp for
        for(int x = 0; x < num/4 * 4; x += 4 ){
            __m256d pos = _mm256_loadu_pd(&(mat->data[x]));
            __m256d neg = _mm256_mul_pd(pos, negOnes);
            _mm256_storeu_pd((__m256d *) &(result->data[x]), _mm256_max_pd(pos, neg));
        }   

        for (int x = num/4 * 4; x < num; x++) {
            double ele = mat->data[x];
            result->data[x] = abs(ele);
        } 
    }

    return 0;
}

/*
 * (OPTIONAL)
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success.
 * Note that the matrix is in row-major order.
 */
int neg_matrix(matrix *result, matrix *mat) {
    // Task 1.5 TODO
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in row-major order.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.5 TODO
    int num = (mat1->rows)*(mat1->cols);
    double *res = result->data;
    double *dat1 = mat1->data;
    double *dat2 = mat2->data;
    
    #pragma omp parallel if (num > 1000)
    {   
        #pragma omp for 
        for(int x = 0; x < num/4 * 4; x += 4){
            __m256d m1 = _mm256_loadu_pd(&(mat1->data[x]));
            __m256d m2 = _mm256_loadu_pd(&(mat2->data[x]));
            __m256d sum = _mm256_add_pd(m1, m2);
            _mm256_storeu_pd((__m256d *) &(result->data[x]), sum);
        }

        for (int x = num/4 * 4; x < num; x++) {
            result->data[x] = (mat1->data[x])+(mat2->data[x]);
        } 
    }
    return 0;
}

/*
 * (OPTIONAL)
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in row-major order.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.5 TODO
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 * You may assume `mat1`'s number of columns is equal to `mat2`'s number of rows.
 * Note that the matrix is in row-major order.
 */

int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.6 TODO
    int row1 = mat1->rows;
    int col1 = mat1->cols; 
    
    

    matrix* trans = transpose(mat2);

    int row2 = trans->rows;
    int col2 = trans->cols;

    //dot product every row of matrix1 with every row of trans
    
    #pragma omp parallel
    {
        #pragma omp for
        for (int r1 = 0; r1 < row1; r1++) {
            for(int r2 = 0; r2 < row2; r2++) {
                double elem = dot(mat1, trans, r1 * col1, r2 * col2);
                set(result, r1, r2, elem);
            }
        }
    } 

    deallocate_matrix(trans);

    return 0;
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 * You may assume `mat` is a square matrix and `pow` is a non-negative integer.
 * Note that the matrix is in row-major order.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    // Task 1.6 TODO
    int pow2;
    if(pow == 1){
         memcpy(result->data, mat->data, result->rows*result->cols*sizeof(double));
        return 0; 
    }
    if(pow == 0){
        for(int x = 0; x<(mat->rows);x++){
            set(result,x,x,1);

        }
        return 0;
    }
    matrix *x;
    allocate_matrix(&x,mat->rows, mat->cols);
    matrix *y;
    allocate_matrix(&y,mat->rows, mat->cols);
    pow_matrix(y,y,0);
    memcpy(x->data, mat->data, result->rows*result->cols*sizeof(double));
    matrix *temp;
    allocate_matrix(&temp,mat->rows, mat->cols);
    while(pow>1){
        if(pow%2==0){
            mul_matrix(temp, x, x);
            memcpy(x->data, temp->data, result->rows*result->cols*sizeof(double));
            pow = pow/2;
        }
        else{
            mul_matrix(temp, x, y);
            memcpy(y->data, temp->data, result->rows*result->cols*sizeof(double));
            mul_matrix(temp, x, x);
            memcpy(x->data, temp->data, result->rows*result->cols*sizeof(double));
            pow = (pow-1)/2;
        }
    }
    mul_matrix(result, x, y);
    return 0;
}



