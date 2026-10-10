#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

// Definition de toute les structures de données utilisées dans le programme
 typedef struct {
    double real;
    double imag;
 } Complex;
 typedef struct{
    double real[4];
    double imag[4];
 } ComplexArray;
 typedef struct {
    Complex alpha;
    Complex beta;
 } Qubit;
 typedef struct {
    Complex coordX;
    Complex coordY;
 } vector;
 typedef struct {
    int rows;
    int cols;
    Complex **data;
 } Matrix;
 typedef struct{
   int num_qubits;
   int states_size;
   Complex *state;
 }QuantumState;

// prototype de certaine fonction pour eviter les warning
void matx_mult_vect(Matrix *matx,Complex *vect,Complex *result);
 //les différentes fonctions pour les opérations sur les nombres complexes
void matx_mult_vec_avx(__m256d matx_real,__m256d matx_imag, __m256d vect_real, __m256d vect_imag,QuantumState *qs,int a,int b,int c,int d){
   __m256d result_real = _mm256_sub_pd(_mm256_mul_pd(matx_real, vect_real), _mm256_mul_pd(matx_imag, vect_imag));
   __m256d result_imag = _mm256_add_pd(_mm256_mul_pd(matx_real, vect_imag), _mm256_mul_pd(matx_imag, vect_real));
   __m256d invert_real = _mm256_permute2f128_pd(matx_real, matx_real, 0x01);
   __m256d invert_imag = _mm256_permute2f128_pd(matx_imag, matx_imag, 0x01);
   __m256d result01_real =_mm256_sub_pd(_mm256_mul_pd(invert_real, vect_real), _mm256_mul_pd(invert_imag, vect_imag)); 
   __m256d result01_imag = _mm256_add_pd(_mm256_mul_pd(invert_real, vect_imag), _mm256_mul_pd(invert_imag, vect_real));
   __m256d result_final_real = _mm256_hadd_pd(result_real, result01_real);
   __m256d result_final_imag = _mm256_hadd_pd(result_imag, result01_imag);
       qs->state[a].imag = ((double*)&result_final_imag)[0];
       qs->state[a].real = ((double*)&result_final_real)[0];
       qs->state[b].real = ((double*)&result_final_real)[1];
       qs->state[b].imag = ((double*)&result_final_imag)[1];
       qs->state[c].real = ((double*)&result_final_real)[3];
       qs->state[c].imag = ((double*)&result_final_imag)[3];
       qs->state[d].real = ((double*)&result_final_real)[2];
       qs->state[d].imag = ((double*)&result_final_imag)[2];
   }

Complex Cpx_add(Complex a, Complex b){
    Complex result;
    result.real =a.real + b.real;
    result.imag = a.imag + b.imag;
    return result;
}
Complex Cpx_mult(Complex a, Complex b){
    Complex result;
    result.real = a.real * b.real - a.imag * b.imag;
    result.imag = a.real * b.imag + a.imag * b.real;
    return result;
}
double Cpx_MagnSq(Complex a){
    return a.real * a.real + a.imag * a.imag;}

 //les différentes fonctions pour les opérations sur les matrices comme la create la liberation de memoire et la multiplication de matrice par un vecteur
 Matrix create_matrix(int rows,int cols){
    Matrix matx;
    Complex init = {0.0, 0.0};
    matx.rows = rows;
    matx.cols = cols;
    matx.data = malloc(rows * sizeof(Complex *));
    for(int i = 0; i < rows; i++){
        matx.data[i] = malloc(cols * sizeof(Complex));
        for(int j = 0; j < cols; j++){
            matx.data[i][j] = init;
        }
    }
    return matx;
 }
 void free_matx(Matrix *matx){
    for(int i = 0;i< matx->rows;i++){
        free(matx->data[i]);
    }
    free(matx->data);
 }
 void generate_pears(int state_size,int step, Matrix *gate,QuantumState *qs){ 
        if(step == 1){
         if(state_size == 2){
             Complex input[2] = {qs->state[0], qs->state[1]};
             Complex result[2];
             matx_mult_vect(gate, input, result);
             qs->state[0] = result[0];
             qs->state[1] = result[1];
             return;
         }

         __m256d real_vec, imag_vec, gate_real_vec, gate_imag_vec;
         ComplexArray Array;
         gate_real_vec = _mm256_set_pd(gate->data[1][1].real, gate->data[1][0].real, gate->data[0][1].real, gate->data[0][0].real);
         gate_imag_vec = _mm256_set_pd(gate->data[1][1].imag, gate->data[1][0].imag, gate->data[0][1].imag, gate->data[0][0].imag);

         for(int base = 0; base < state_size; base += 4){
             int a = base;
             int b = base + 1;
             int c = base + 2;
             int d = base + 3;
             Array.real[0] = qs->state[a].real;
             Array.imag[0] = qs->state[a].imag;
             Array.real[1] = qs->state[b].real;
             Array.imag[1] = qs->state[b].imag;
             Array.real[2] = qs->state[c].real;
             Array.imag[2] = qs->state[c].imag;
             Array.real[3] = qs->state[d].real;
             Array.imag[3] = qs->state[d].imag;
             real_vec = _mm256_loadu_pd(Array.real);
             imag_vec = _mm256_loadu_pd(Array.imag);
             matx_mult_vec_avx(gate_real_vec, gate_imag_vec, real_vec, imag_vec, qs, a, b, c, d);
         }
         return;
        }

        __m256d real_vec, imag_vec, gate_real_vec, gate_imag_vec;
        ComplexArray Array;
        for(int base=0;base<state_size;base+=2*step){
        for(int ofset=0;ofset<step;ofset+=2){
            int ofprim=ofset+1;
            int a=base+ofset;
            int b = a +step;
            int c = base + ofprim;
            int d = c + step;
           Array.real[0] = qs->state[a].real;
           Array.imag[0] = qs->state[a].imag;
           Array.real[1] = qs->state[b].real;
           Array.imag[1] = qs->state[b].imag;
           Array.real[2] = qs->state[c].real;
           Array.imag[2] = qs->state[c].imag;
           Array.real[3] = qs->state[d].real;
           Array.imag[3] = qs->state[d].imag;
           real_vec = _mm256_loadu_pd(Array.real);
           imag_vec = _mm256_loadu_pd(Array.imag);
           gate_real_vec = _mm256_set_pd(gate->data[1][1].real, gate->data[1][0].real, gate->data[0][1].real, gate->data[0][0].real);
           gate_imag_vec = _mm256_set_pd(gate->data[1][1].imag, gate->data[1][0].imag, gate->data[0][1].imag, gate->data[0][0].imag); 

            matx_mult_vec_avx(gate_real_vec, gate_imag_vec, real_vec, imag_vec, qs, a, b, c, d);

            
         

        }
       }
 }
 void matx_mult_vect(Matrix *matx,Complex *vect,Complex *result){
    result[0] = Cpx_add(Cpx_mult(matx->data[0][0], vect[0]), Cpx_mult(matx->data[0][1], vect[1]));
    result[1] = Cpx_add(Cpx_mult(matx->data[1][0], vect[0]), Cpx_mult(matx->data[1][1], vect[1]));

 }
 //les différentes fonctions pour créer les matrices de portes quantiques
 Matrix create_X(){
    Matrix X = create_matrix(2, 2);
    X.data[0][1] = (Complex){1.0, 0.0};
    X.data[1][0] = (Complex){1.0, 0.0};
    return X;
 }
 Matrix create_H(){
    Matrix H = create_matrix(2, 2);
    double inv_sqrt2 = 1.0 / sqrt(2.0);
    H.data[0][0] = (Complex){inv_sqrt2, 0.0};
    H.data[0][1] = (Complex){inv_sqrt2, 0.0};
    H.data[1][0] = (Complex){inv_sqrt2, 0.0};
    H.data[1][1] = (Complex){-inv_sqrt2, 0.0};
    return H;
 }
 // Function to create a quantum state with a given number of qubits
 QuantumState create_quantum_state(int num_qubits,int o){
    QuantumState qs;
      qs.num_qubits = num_qubits;
    qs.states_size = 1;
    for(int i = 0; i < qs.num_qubits; i++){
         qs.states_size *= 2;
    }
    if(o<0 || o>qs.states_size-1){
      printf("Une erreur s'est produite : le second argument representant l'etat initial doit etre entre 0 et %d",qs.states_size);
      return (QuantumState){1,1,NULL};
    }
    qs.state= malloc(qs.states_size * sizeof(Complex));
    for(int i = 0; i < qs.states_size; i++){
        qs.state[i].real = 0.0;
        qs.state[i].imag = 0.0;
    }
    qs.state[o]= (Complex){1.0, 0.0};

    return qs;
 }
 int Measur_Quant(QuantumState *qs){
   double cumul=0.0;
   int i;
   double random=(double)rand()/RAND_MAX;
   for(i =0;i<qs->states_size;i++){
       cumul +=Cpx_MagnSq(qs->state[i]);
       if(random<=cumul){
         QuantumState rs = create_quantum_state(qs->num_qubits,i);
         free(qs->state);
         *qs=rs;
          return i;
       }
    }
     return -1;
 }
 //fonction qui libère la mémoire allouée pour l'état quantique
 void free_quantum_state(QuantumState *qs)
{
    free(qs->state);
}
void Quant_Prob(QuantumState *q){
   for(int i =0;i<q->states_size;i++){
     printf("La probabilite d'apparition du State %d est %f%% \n",i,Cpx_MagnSq(q->state[i])*100); 
   }
}
// Function to apply a quantum gate to a specific qubit in the quantum state
void apply_gate(QuantumState *qs, Matrix *gate,int target){
   int st = (qs->num_qubits-1-target);
   int step=1<<st;
   
   generate_pears(qs->states_size,step,gate,qs);
}

//le main mais pour le moment on fait juste des test avec
int main(){



    srand((unsigned int)time(NULL));

    Matrix H = create_H();

    // ==========================================
    // STRESS TEST
    // ==========================================
    int n;
    printf("ENTRER LE NOMBRE DE QUBITS : ");
    scanf("%d", &n);
    size_t states = (size_t)1 << n;
size_t memory = states * sizeof(Complex);

printf("Qubits : %d\n", n);
printf("Etats  : %zu\n", states);
printf("Memoire necessaire : %.2f Go\n",
       (double)memory / (1024.0 * 1024.0 * 1024.0));

    printf("\n====================================\n");
    printf("          STRESS TEST\n");
    printf("====================================\n\n");

    printf("Creation d'un etat de %d qubits...\n", n);

    clock_t start = clock();

    QuantumState qs = create_quantum_state(n, 0);

    printf("Nombre de qubits : %d\n", qs.num_qubits);
    printf("Nombre d'etats   : %d\n", qs.states_size);

    // ==========================================
    // APPLICATION DES PORTES H
    // ==========================================

    printf("\nApplication de %d portes H...\n", n);

    for(int i = 0; i < n; i++){
        int step = 1 << (n - 1 - i);
        clock_t gate_start = clock();
        apply_gate(&qs, &H, i);
        clock_t gate_end = clock();
        double gate_elapsed = (double)(gate_end - gate_start) / CLOCKS_PER_SEC;
        printf("Porte H appliquee sur le qubit %d en %.6f secondes step=%d\n", i, gate_elapsed, step);
    }

    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    printf("\n====================================\n");
    printf("             RESULTAT\n");
    printf("====================================\n\n");

    printf("Qubits       : %d\n", qs.num_qubits);
    printf("Etats        : %d\n", qs.states_size);
    printf("Temps        : %.6f secondes\n", elapsed);

    // ==========================================
    // VERIFICATION
    // ==========================================

    printf("\nQuelques amplitudes :\n");

    for(int i = 0; i < 10 && i < qs.states_size; i++){
        printf(
            "state[%d] = %.6f + %.6fi\n",
            i,
            qs.state[i].real,
            qs.state[i].imag
        );
    }

    // ==========================================
    // LIBERATION
    // ==========================================

    free_quantum_state(&qs);
    free_matx(&H);

    printf("\n====================================\n");
    printf("              FIN\n");
    printf("====================================\n");

    return 0;
}
