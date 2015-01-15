#include <glm/glm.hpp>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

typedef unsigned int uint;

// Constantes para la particion del espacio

// Estos valores son las dimensiones del MBR de los vertices de un boid
// [0.03, 0.05, 0.2]
#define kCellSizeX 0.031
#define kCellSizeY 0.031
#define kCellSizeZ 0.031

// TODO Revisa estos valores, trata de asegurar que boidPos
// te devuelva posiciones <= 64

#define kGridSizeX 64
#define kGridSizeY 64
#define kGridSizeZ 64

#define kNumCells (kGridSizeX * kGridSizeY * kGridSizeZ)

// Cambiar el origen a (-1,-1,-1) ?

#define kOriginX -10.0
#define kOriginY -10.0
#define kOriginZ -10.0

__device__ int GPU_globalindex() {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int index = x + y * blockDim.x * gridDim.x;
    return index;
}

__device__ int3 boidPos(glm::vec3 p) {
    int3 gridPos;
    gridPos.x = floor((p.x - kOriginX) / kCellSizeX);
    gridPos.y = floor((p.y - kOriginY) / kCellSizeY);
    gridPos.z = floor((p.z - kOriginZ) / kCellSizeZ);
    return gridPos;
}

__device__ uint boidPosToHash(int3 gridPos) {
    gridPos.x = gridPos.x & (kGridSizeX-1);
    gridPos.y = gridPos.y & (kGridSizeY-1);
    gridPos.z = gridPos.z & (kGridSizeZ-1);
    // TODO Probar con la curva Z, deberia mejorar los cache hits
    return gridPos.z * kGridSizeY * kGridSizeX + gridPos.y * kGridSizeX + gridPos.x;
}

// Paso 1 Hash mapping
// Calcula el hash de cada boid
__global__ void hashBoidsKernel(uint   *boids_hash,
               uint   *boids_index,
               glm::vec3 *pos,
               uint    num_boids) {
    uint index = GPU_globalindex();
    if (index >= num_boids) return;
    glm::vec3 p = pos[index];
    int3 gridPos = boidPos(p);
    uint hash = boidPosToHash(gridPos);
    boids_hash[index] = hash;
    boids_index[index] = index;
}

// Paso 2 Sorting
// Esto ordena los pares <CELL ID, INDIVIDUAL ID> por CELL ID
void sortBoidsByHash(uint *boids_hash, uint *boids_index, uint num_boids) {
    thrust::sort_by_key(thrust::device_ptr<uint>(boids_hash),
            thrust::device_ptr<uint>(boids_hash + num_boids),
            thrust::device_ptr<uint>(boids_index));
}

// Paso 3 Data reordering & Cells Counting
// Reordena los datos (posicion, orientacion) segun la ordenacion
// del paso 2 y encuentra el rango de todos los boids que se
// encuentran en una celda data
__global__ void reorderData(uint   *cell_start,
                            uint   *cell_end,
                            glm::vec3 *sorted_pos,
                            glm::vec3 *sorted_dir,
                            uint   *boids_hash,
                            uint   *boids_index,
                            glm::vec3 *old_pos,
                            glm::vec3 *old_dir,
                            uint    num_boids) {
    extern __shared__ uint sharedHash[];
    uint index = GPU_globalindex();
    uint hash;

    if (index < num_boids) {
        hash = boids_hash[index];

        sharedHash[threadIdx.x+1] = hash;

        if (index > 0 && threadIdx.x == 0) {
            sharedHash[0] = boids_hash[index-1];
        }
    }

    __syncthreads();

    if (index < num_boids) {
        if (index == 0 || hash != sharedHash[threadIdx.x]) {
            cell_start[hash] = index;

            if (index > 0)
                cell_end[sharedHash[threadIdx.x]] = index;
        }

        if (index == num_boids - 1) {
            cell_end[hash] = index + 1;
        }

        uint sortedIndex = boids_index[index];
        glm::vec3 pos = old_pos[sortedIndex];
        glm::vec3 dir = old_dir[sortedIndex];
        sorted_pos[index] = pos;
        sorted_dir[index] = dir;
    }
}

void hashBoids(uint  *boids_hash, uint  *boids_index,
        glm::vec3 *pos, int num_boids) {
    dim3 grids(num_boids, 1);
    dim3 threads(1, 1);
    hashBoidsKernel<<< grids, threads>>>(boids_hash, boids_index, pos, num_boids);
    getLastCudaError("hashBoidsKernel failed");
}

void reorderDataP(uint  *cell_start,
        uint  *cell_end,
        glm::vec3 *sorted_pos,
        glm::vec3 *sorted_dir,
        uint  *boids_hash,
        uint  *boids_index,
        glm::vec3 *pos,
        glm::vec3 *dir,
        uint   num_boids) {
    uint numThreads, numBlocks;
    numThreads = min(256, num_boids);
    numBlocks = (num_boids + numThreads - 1) / numThreads;

    checkCudaErrors(cudaMemset(cell_start, 0xffffffff, kNumCells * sizeof(uint)));

    uint smemSize = sizeof(uint) * (numThreads+1);
    reorderData<<< numBlocks, numThreads, smemSize>>>(
            cell_start,
            cell_end,
            sorted_pos,
            sorted_dir,
            boids_hash,
            boids_index,
            pos,
            dir,
            num_boids);
    getLastCudaError("reorderData kernel failed");
}
