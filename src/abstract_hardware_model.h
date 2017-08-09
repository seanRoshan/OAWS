// Copyright (c) 2009-2011, Tor M. Aamodt, Inderpreet Singh,
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// Neither the name of The University of British Columbia nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef ABSTRACT_HARDWARE_MODEL_INCLUDED
#define ABSTRACT_HARDWARE_MODEL_INCLUDED

// Forward declarations
class gpgpu_sim;
class kernel_info_t;
class MPRB;
class DRSVR;

class warp_inst_t;

//Set a hard limit of 32 CTAs per shader [cuda only has 8]
#define MAX_CTA_PER_SHADER 32
#define MAX_BARRIERS_PER_CTA 16

enum _memory_space_t {
   undefined_space=0,
   reg_space,
   local_space,
   shared_space,
   param_space_unclassified,
   param_space_kernel,  /* global to all threads in a kernel : read-only */
   param_space_local,   /* local to a thread : read-writable */
   const_space,
   tex_space,
   surf_space,
   global_space,
   generic_space,
   instruction_space
};


enum FuncCache
{
  FuncCachePreferNone = 0,
  FuncCachePreferShared = 1,
  FuncCachePreferL1 = 2
};


#ifdef __cplusplus

#include <string.h>
#include <stdio.h>
#include <csignal>
#include <bitset>
#include <math.h>
#include <sstream>


typedef unsigned long long new_addr_type;
typedef unsigned address_type;
typedef unsigned addr_t;

// the following are operations the timing model can see

enum uarch_op_t {
   NO_OP=-1,
   ALU_OP=1,
   SFU_OP,
   ALU_SFU_OP,
   LOAD_OP,
   STORE_OP,
   BRANCH_OP,
   BARRIER_OP,
   MEMORY_BARRIER_OP,
   CALL_OPS,
   RET_OPS
};
typedef enum uarch_op_t op_type;


enum uarch_bar_t {
   NOT_BAR=-1,
   SYNC=1,
   ARRIVE,
   RED
};
typedef enum uarch_bar_t barrier_type;

enum uarch_red_t {
   NOT_RED=-1,
   POPC_RED=1,
   AND_RED,
   OR_RED
};
typedef enum uarch_red_t reduction_type;


enum uarch_operand_type_t {
	UN_OP=-1,
    INT_OP,
    FP_OP
};

typedef enum uarch_operand_type_t types_of_operands;

enum special_operations_t {
    OTHER_OP,
    INT__OP,
	INT_MUL24_OP,
	INT_MUL32_OP,
	INT_MUL_OP,
    INT_DIV_OP,
    FP_MUL_OP,
    FP_DIV_OP,
    FP__OP,
	FP_SQRT_OP,
	FP_LG_OP,
	FP_SIN_OP,
	FP_EXP_OP
};


typedef enum special_operations_t special_ops; // Required to identify for the power model
enum operation_pipeline_t {
    UNKOWN_OP,
    SP__OP,
    SFU__OP,
    MEM__OP
};
typedef enum operation_pipeline_t operation_pipeline;
enum mem_operation_t {
    NOT_TEX,
    TEX
};
typedef enum mem_operation_t mem_operation;

enum _memory_op_t {
	no_memory_op = 0,
	memory_load,
	memory_store
};

#include <bitset>
#include <list>
#include <vector>
#include <assert.h>
#include <stdlib.h>
#include <map>
#include <deque>

#if !defined(__VECTOR_TYPES_H__)
struct dim3 {
   unsigned int x, y, z;
};
#endif

void increment_x_then_y_then_z( dim3 &i, const dim3 &bound);

class kernel_info_t {
public:
//   kernel_info_t()
//   {
//      m_valid=false;
//      m_kernel_entry=NULL;
//      m_uid=0;
//      m_num_cores_running=0;
//      m_param_mem=NULL;
//   }
   kernel_info_t( dim3 gridDim, dim3 blockDim, class function_info *entry );
   ~kernel_info_t();

   void inc_running() { m_num_cores_running++; }
   void dec_running()
   {
       assert( m_num_cores_running > 0 );
       m_num_cores_running--;
   }
   bool running() const { return m_num_cores_running>0; }
   bool done() const
   {
       return no_more_ctas_to_run() && !running();
   }
   class function_info *entry() { return m_kernel_entry; }
   const class function_info *entry() const { return m_kernel_entry; }

   size_t num_blocks() const
   {
      return m_grid_dim.x * m_grid_dim.y * m_grid_dim.z;
   }

   size_t threads_per_cta() const
   {
      return m_block_dim.x * m_block_dim.y * m_block_dim.z;
   }

   dim3 get_grid_dim() const { return m_grid_dim; }
   dim3 get_cta_dim() const { return m_block_dim; }

   void increment_cta_id()
   {
      increment_x_then_y_then_z(m_next_cta,m_grid_dim);
      m_next_tid.x=0;
      m_next_tid.y=0;
      m_next_tid.z=0;
   }
   dim3 get_next_cta_id() const { return m_next_cta; }
   bool no_more_ctas_to_run() const
   {
      return (m_next_cta.x >= m_grid_dim.x || m_next_cta.y >= m_grid_dim.y || m_next_cta.z >= m_grid_dim.z );
   }

   void increment_thread_id() { increment_x_then_y_then_z(m_next_tid,m_block_dim); }
   dim3 get_next_thread_id_3d() const  { return m_next_tid; }
   unsigned get_next_thread_id() const
   {
      return m_next_tid.x + m_block_dim.x*m_next_tid.y + m_block_dim.x*m_block_dim.y*m_next_tid.z;
   }
   bool more_threads_in_cta() const
   {
      return m_next_tid.z < m_block_dim.z && m_next_tid.y < m_block_dim.y && m_next_tid.x < m_block_dim.x;
   }
   unsigned get_uid() const { return m_uid; }
   std::string name() const;

   std::list<class ptx_thread_info *> &active_threads() { return m_active_threads; }
   class memory_space *get_param_memory() { return m_param_mem; }

private:
   kernel_info_t( const kernel_info_t & ); // disable copy constructor
   void operator=( const kernel_info_t & ); // disable copy operator

   class function_info *m_kernel_entry;

   unsigned m_uid;
   static unsigned m_next_uid;

   dim3 m_grid_dim;
   dim3 m_block_dim;
   dim3 m_next_cta;
   dim3 m_next_tid;

   unsigned m_num_cores_running;

   std::list<class ptx_thread_info *> m_active_threads;
   class memory_space *m_param_mem;
};

struct core_config {
    core_config()
    {
        m_valid = false;
        num_shmem_bank=16;
        shmem_limited_broadcast = false;
        gpgpu_shmem_sizeDefault=(unsigned)-1;
        gpgpu_shmem_sizePrefL1=(unsigned)-1;
        gpgpu_shmem_sizePrefShared=(unsigned)-1;
    }
    virtual void init() = 0;

    bool m_valid;
    unsigned warp_size;

    // off-chip memory request architecture parameters
    int gpgpu_coalesce_arch;

    // shared memory bank conflict checking parameters
    bool shmem_limited_broadcast;
    static const address_type WORD_SIZE=4;
    unsigned num_shmem_bank;
    unsigned shmem_bank_func(address_type addr) const
    {
        return ((addr/WORD_SIZE) % num_shmem_bank);
    }
    unsigned mem_warp_parts;
    mutable unsigned gpgpu_shmem_size;
    unsigned gpgpu_shmem_sizeDefault;
    unsigned gpgpu_shmem_sizePrefL1;
    unsigned gpgpu_shmem_sizePrefShared;

    // texture and constant cache line sizes (used to determine number of memory accesses)
    unsigned gpgpu_cache_texl1_linesize;
    unsigned gpgpu_cache_constl1_linesize;

    // DRSVR data cache set info
    unsigned gpgpu_cache_dl1_setnumber;
    unsigned gpgpu_cache_dl1_setnumberlog2;

	unsigned gpgpu_max_insn_issue_per_warp;
};

// bounded stack that implements simt reconvergence using pdom mechanism from MICRO'07 paper
const unsigned MAX_WARP_SIZE = 32;
typedef std::bitset<MAX_WARP_SIZE> active_mask_t;
#define MAX_WARP_SIZE_SIMT_STACK  MAX_WARP_SIZE
typedef std::bitset<MAX_WARP_SIZE_SIMT_STACK> simt_mask_t;
typedef std::vector<address_type> addr_vector_t;

class simt_stack {
public:
    simt_stack( unsigned wid,  unsigned warpSize);

    void reset();
    void launch( address_type start_pc, const simt_mask_t &active_mask );
    void update( simt_mask_t &thread_done, addr_vector_t &next_pc, address_type recvg_pc, op_type next_inst_op,unsigned next_inst_size, address_type next_inst_pc );

    const simt_mask_t &get_active_mask() const;
    void     get_pdom_stack_top_info( unsigned *pc, unsigned *rpc ) const;
    unsigned get_rp() const;
    void     print(FILE*fp) const;
    void     print2() const;
    unsigned get_m_stackSize() const;

protected:
    unsigned m_warp_id;
    unsigned m_warp_size;
    unsigned m_sm_id;

    enum stack_entry_type {
        STACK_ENTRY_TYPE_NORMAL = 0,
        STACK_ENTRY_TYPE_CALL
    };

    struct simt_stack_entry {
        address_type m_pc;
        unsigned int m_calldepth;
        simt_mask_t m_active_mask;
        address_type m_recvg_pc;
        unsigned long long m_branch_div_cycle;
        stack_entry_type m_type;
        simt_stack_entry() :
            m_pc(-1), m_calldepth(0), m_active_mask(), m_recvg_pc(-1), m_branch_div_cycle(0), m_type(STACK_ENTRY_TYPE_NORMAL) { };
    };

    std::deque<simt_stack_entry> m_stack;
};

#define GLOBAL_HEAP_START 0x80000000
   // start allocating from this address (lower values used for allocating globals in .ptx file)
#define SHARED_MEM_SIZE_MAX (64*1024)
#define LOCAL_MEM_SIZE_MAX (8*1024)
#define MAX_STREAMING_MULTIPROCESSORS 64
#define MAX_THREAD_PER_SM 2048
#define TOTAL_LOCAL_MEM_PER_SM (MAX_THREAD_PER_SM*LOCAL_MEM_SIZE_MAX)
#define TOTAL_SHARED_MEM (MAX_STREAMING_MULTIPROCESSORS*SHARED_MEM_SIZE_MAX)
#define TOTAL_LOCAL_MEM (MAX_STREAMING_MULTIPROCESSORS*MAX_THREAD_PER_SM*LOCAL_MEM_SIZE_MAX)
#define SHARED_GENERIC_START (GLOBAL_HEAP_START-TOTAL_SHARED_MEM)
#define LOCAL_GENERIC_START (SHARED_GENERIC_START-TOTAL_LOCAL_MEM)
#define STATIC_ALLOC_LIMIT (GLOBAL_HEAP_START - (TOTAL_LOCAL_MEM+TOTAL_SHARED_MEM))

#if !defined(__CUDA_RUNTIME_API_H__)

enum cudaChannelFormatKind {
   cudaChannelFormatKindSigned,
   cudaChannelFormatKindUnsigned,
   cudaChannelFormatKindFloat
};

struct cudaChannelFormatDesc {
   int                        x;
   int                        y;
   int                        z;
   int                        w;
   enum cudaChannelFormatKind f;
};

struct cudaArray {
   void *devPtr;
   int devPtr32;
   struct cudaChannelFormatDesc desc;
   int width;
   int height;
   int size; //in bytes
   unsigned dimensions;
};

enum cudaTextureAddressMode {
   cudaAddressModeWrap,
   cudaAddressModeClamp
};

enum cudaTextureFilterMode {
   cudaFilterModePoint,
   cudaFilterModeLinear
};

enum cudaTextureReadMode {
   cudaReadModeElementType,
   cudaReadModeNormalizedFloat
};

struct textureReference {
   int                           normalized;
   enum cudaTextureFilterMode    filterMode;
   enum cudaTextureAddressMode   addressMode[3];
   struct cudaChannelFormatDesc  channelDesc;
};

#endif

// Struct that record other attributes in the textureReference declaration
// - These attributes are passed thru __cudaRegisterTexture()
struct textureReferenceAttr {
    const struct textureReference *m_texref;
    int m_dim;
    enum cudaTextureReadMode m_readmode;
    int m_ext;
    textureReferenceAttr(const struct textureReference *texref,
                         int dim,
                         enum cudaTextureReadMode readmode,
                         int ext)
    : m_texref(texref), m_dim(dim), m_readmode(readmode), m_ext(ext)
    {  }
};

class gpgpu_functional_sim_config
{
public:
    void reg_options(class OptionParser * opp);

    void ptx_set_tex_cache_linesize(unsigned linesize);

    unsigned get_forced_max_capability() const { return m_ptx_force_max_capability; }
    bool convert_to_ptxplus() const { return m_ptx_convert_to_ptxplus; }
    bool use_cuobjdump() const { return m_ptx_use_cuobjdump; }
    bool experimental_lib_support() const { return m_experimental_lib_support; }

    int         get_ptx_inst_debug_to_file() const { return g_ptx_inst_debug_to_file; }
    const char* get_ptx_inst_debug_file() const  { return g_ptx_inst_debug_file; }
    int         get_ptx_inst_debug_thread_uid() const { return g_ptx_inst_debug_thread_uid; }
    unsigned    get_texcache_linesize() const { return m_texcache_linesize; }

private:
    // PTX options
    int m_ptx_convert_to_ptxplus;
    int m_ptx_use_cuobjdump;
    int m_experimental_lib_support;
    unsigned m_ptx_force_max_capability;

    int   g_ptx_inst_debug_to_file;
    char* g_ptx_inst_debug_file;
    int   g_ptx_inst_debug_thread_uid;

    unsigned m_texcache_linesize;
};

class gpgpu_t {
public:
    gpgpu_t( const gpgpu_functional_sim_config &config );
    void* gpu_malloc( size_t size );
    void* gpu_mallocarray( size_t count );
    void  gpu_memset( size_t dst_start_addr, int c, size_t count );
    void  memcpy_to_gpu( size_t dst_start_addr, const void *src, size_t count );
    void  memcpy_from_gpu( void *dst, size_t src_start_addr, size_t count );
    void  memcpy_gpu_to_gpu( size_t dst, size_t src, size_t count );

    class memory_space *get_global_memory() { return m_global_mem; }
    class memory_space *get_tex_memory() { return m_tex_mem; }
    class memory_space *get_surf_memory() { return m_surf_mem; }

    void gpgpu_ptx_sim_bindTextureToArray(const struct textureReference* texref, const struct cudaArray* array);
    void gpgpu_ptx_sim_bindNameToTexture(const char* name, const struct textureReference* texref, int dim, int readmode, int ext);
    const char* gpgpu_ptx_sim_findNamefromTexture(const struct textureReference* texref);

    const struct textureReference* get_texref(const std::string &texname) const
    {
        std::map<std::string, const struct textureReference*>::const_iterator t=m_NameToTextureRef.find(texname);
        assert( t != m_NameToTextureRef.end() );
        return t->second;
    }
    const struct cudaArray* get_texarray( const struct textureReference *texref ) const
    {
        std::map<const struct textureReference*,const struct cudaArray*>::const_iterator t=m_TextureRefToCudaArray.find(texref);
        assert(t != m_TextureRefToCudaArray.end());
        return t->second;
    }
    const struct textureInfo* get_texinfo( const struct textureReference *texref ) const
    {
        std::map<const struct textureReference*, const struct textureInfo*>::const_iterator t=m_TextureRefToTexureInfo.find(texref);
        assert(t != m_TextureRefToTexureInfo.end());
        return t->second;
    }

    const struct textureReferenceAttr* get_texattr( const struct textureReference *texref ) const
    {
        std::map<const struct textureReference*, const struct textureReferenceAttr*>::const_iterator t=m_TextureRefToAttribute.find(texref);
        assert(t != m_TextureRefToAttribute.end());
        return t->second;
    }

    const gpgpu_functional_sim_config &get_config() const { return m_function_model_config; }
    FILE* get_ptx_inst_debug_file() { return ptx_inst_debug_file; }

protected:
    const gpgpu_functional_sim_config &m_function_model_config;
    FILE* ptx_inst_debug_file;

    class memory_space *m_global_mem;
    class memory_space *m_tex_mem;
    class memory_space *m_surf_mem;

    unsigned long long m_dev_malloc;

    std::map<std::string, const struct textureReference*> m_NameToTextureRef;
    std::map<const struct textureReference*,const struct cudaArray*> m_TextureRefToCudaArray;
    std::map<const struct textureReference*, const struct textureInfo*> m_TextureRefToTexureInfo;
    std::map<const struct textureReference*, const struct textureReferenceAttr*> m_TextureRefToAttribute;
};

struct gpgpu_ptx_sim_kernel_info
{
   // Holds properties of the kernel (Kernel's resource use).
   // These will be set to zero if a ptxinfo file is not present.
   int lmem;
   int smem;
   int cmem;
   int regs;
   unsigned ptx_version;
   unsigned sm_target;
};

struct gpgpu_ptx_sim_arg {
   gpgpu_ptx_sim_arg() { m_start=NULL; }
   gpgpu_ptx_sim_arg(const void *arg, size_t size, size_t offset)
   {
      m_start=arg;
      m_nbytes=size;
      m_offset=offset;
   }
   const void *m_start;
   size_t m_nbytes;
   size_t m_offset;
};

typedef std::list<gpgpu_ptx_sim_arg> gpgpu_ptx_sim_arg_list_t;

class memory_space_t {
public:
   memory_space_t() { m_type = undefined_space; m_bank=0; }
   memory_space_t( const enum _memory_space_t &from ) { m_type = from; m_bank = 0; }
   bool operator==( const memory_space_t &x ) const { return (m_bank == x.m_bank) && (m_type == x.m_type); }
   bool operator!=( const memory_space_t &x ) const { return !(*this == x); }
   bool operator<( const memory_space_t &x ) const
   {
      if(m_type < x.m_type)
         return true;
      else if(m_type > x.m_type)
         return false;
      else if( m_bank < x.m_bank )
         return true;
      return false;
   }
   enum _memory_space_t get_type() const { return m_type; }
   unsigned get_bank() const { return m_bank; }
   void set_bank( unsigned b ) { m_bank = b; }
   bool is_const() const { return (m_type == const_space) || (m_type == param_space_kernel); }
   bool is_local() const { return (m_type == local_space) || (m_type == param_space_local); }
   bool is_global() const { return (m_type == global_space); }

private:
   enum _memory_space_t m_type;
   unsigned m_bank; // n in ".const[n]"; note .const == .const[0] (see PTX 2.1 manual, sec. 5.1.3)
};

const unsigned MAX_MEMORY_ACCESS_SIZE = 128;
typedef std::bitset<MAX_MEMORY_ACCESS_SIZE> mem_access_byte_mask_t;
#define NO_PARTIAL_WRITE (mem_access_byte_mask_t())

#define MEM_ACCESS_TYPE_TUP_DEF \
MA_TUP_BEGIN( mem_access_type ) \
   MA_TUP( GLOBAL_ACC_R ), \
   MA_TUP( LOCAL_ACC_R ), \
   MA_TUP( CONST_ACC_R ), \
   MA_TUP( TEXTURE_ACC_R ), \
   MA_TUP( GLOBAL_ACC_W ), \
   MA_TUP( LOCAL_ACC_W ), \
   MA_TUP( L1_WRBK_ACC ), \
   MA_TUP( L2_WRBK_ACC ), \
   MA_TUP( INST_ACC_R ), \
   MA_TUP( L1_WR_ALLOC_R ), \
   MA_TUP( L2_WR_ALLOC_R ), \
   MA_TUP( NUM_MEM_ACCESS_TYPE ) \
MA_TUP_END( mem_access_type )

#define MA_TUP_BEGIN(X) enum X {
#define MA_TUP(X) X
#define MA_TUP_END(X) };
MEM_ACCESS_TYPE_TUP_DEF
#undef MA_TUP_BEGIN
#undef MA_TUP
#undef MA_TUP_END

const char * mem_access_type_str(enum mem_access_type access_type);

enum cache_operator_type {
    CACHE_UNDEFINED,

    // loads
    CACHE_ALL,          // .ca
    CACHE_LAST_USE,     // .lu
    CACHE_VOLATILE,     // .cv

    // loads and stores
    CACHE_STREAMING,    // .cs
    CACHE_GLOBAL,       // .cg

    // stores
    CACHE_WRITE_BACK,   // .wb
    CACHE_WRITE_THROUGH // .wt
};

class mem_access_t {
public:

   mem_access_t() { init(); }

   mem_access_t( mem_access_type type,
                 new_addr_type address,
                 unsigned size,
                 bool wr )
   {
       init();
       m_type = type;
       m_addr = address;
       m_req_size = size;
       m_write = wr;
   }

   mem_access_t( mem_access_type type,
                 new_addr_type address,
                 unsigned size,
                 bool wr,
                 const active_mask_t &active_mask,
                 const mem_access_byte_mask_t &byte_mask )
    : m_warp_mask(active_mask), m_byte_mask(byte_mask)
   {
      init();
      m_type = type;
      m_addr = address;
      m_req_size = size;
      m_write = wr;
   }

   new_addr_type get_addr() const { return m_addr; }
   void set_addr(new_addr_type addr) {m_addr=addr;}
   unsigned get_size() const { return m_req_size; }
   const active_mask_t &get_warp_mask() const { return m_warp_mask; }
   bool is_write() const { return m_write; }
   enum mem_access_type get_type() const { return m_type; }
   mem_access_byte_mask_t get_byte_mask() const { return m_byte_mask; }




   // MPRB ADDED

   void mprb_set_warp_id (unsigned warp_id){
       mprb_wid = warp_id;
   }

   unsigned mprb_get_warp_id (){
       return mprb_wid;
   }

   void print(FILE *fp) const
   {
       fprintf(fp,"addr=0x%llx, %s, size=%u, ", m_addr, m_write?"store":"load ", m_req_size );
       switch(m_type) {
       case GLOBAL_ACC_R:  fprintf(fp,"GLOBAL_R"); break;
       case LOCAL_ACC_R:   fprintf(fp,"LOCAL_R "); break;
       case CONST_ACC_R:   fprintf(fp,"CONST   "); break;
       case TEXTURE_ACC_R: fprintf(fp,"TEXTURE "); break;
       case GLOBAL_ACC_W:  fprintf(fp,"GLOBAL_W"); break;
       case LOCAL_ACC_W:   fprintf(fp,"LOCAL_W "); break;
       case L2_WRBK_ACC:   fprintf(fp,"L2_WRBK "); break;
       case INST_ACC_R:    fprintf(fp,"INST    "); break;
       case L1_WRBK_ACC:   fprintf(fp,"L1_WRBK "); break;
       default:            fprintf(fp,"unknown "); break;
       }
   }

    void print2() const
    {
        //printf("addr=0x%llx, %s, size=%u, ", m_addr, m_write?"store":"load ", m_req_size );
        printf("warp_id=%u ; addr=%u, %s, size=%u, ", mprb_wid, m_addr, m_write?"store":"load ", m_req_size );
        switch(m_type) {
            case GLOBAL_ACC_R:  printf("GLOBAL_R"); break;
            case LOCAL_ACC_R:   printf("LOCAL_R "); break;
            case CONST_ACC_R:   printf("CONST   "); break;
            case TEXTURE_ACC_R: printf("TEXTURE "); break;
            case GLOBAL_ACC_W:  printf("GLOBAL_W"); break;
            case LOCAL_ACC_W:   printf("LOCAL_W "); break;
            case L2_WRBK_ACC:   printf("L2_WRBK "); break;
            case INST_ACC_R:    printf("INST    "); break;
            case L1_WRBK_ACC:   printf("L1_WRBK "); break;
            default:            printf("unknown "); break;
        }
    }

    void print2_toFile(FILE* statFile) const
    {
        //printf("addr=0x%llx, %s, size=%u, ", m_addr, m_write?"store":"load ", m_req_size );
        fprintf(statFile, "warp_id=%u ; addr=%u, %s, size=%u, ", mprb_wid, m_addr, m_write?"store":"load ", m_req_size );
        switch(m_type) {
            case GLOBAL_ACC_R:  fprintf(statFile,"GLOBAL_R"); break;
            case LOCAL_ACC_R:   fprintf(statFile,"LOCAL_R "); break;
            case CONST_ACC_R:   fprintf(statFile,"CONST   "); break;
            case TEXTURE_ACC_R: fprintf(statFile,"TEXTURE "); break;
            case GLOBAL_ACC_W:  fprintf(statFile,"GLOBAL_W"); break;
            case LOCAL_ACC_W:   fprintf(statFile,"LOCAL_W "); break;
            case L2_WRBK_ACC:   fprintf(statFile,"L2_WRBK "); break;
            case INST_ACC_R:    fprintf(statFile,"INST    "); break;
            case L1_WRBK_ACC:   fprintf(statFile,"L1_WRBK "); break;
            default:            fprintf(statFile,"unknown "); break;
        }
    }

private:
   void init()
   {
      m_uid=++sm_next_access_uid;
      m_addr=0;
      m_req_size=0;
      mprb_wid=0;
      //mprb_issue_cycle_time = 0;
   }

   unsigned mprb_wid;

   unsigned      m_uid;
   new_addr_type m_addr;     // request address
   bool          m_write;
   unsigned      m_req_size; // bytes
   mem_access_type m_type;
   active_mask_t m_warp_mask;
   mem_access_byte_mask_t m_byte_mask;

   static unsigned sm_next_access_uid;
};

class mem_fetch;

class mem_fetch_interface {
public:
    virtual bool full( unsigned size, bool write ) const = 0;
    virtual void push( mem_fetch *mf ) = 0;
};

class mem_fetch_allocator {
public:
    virtual mem_fetch *alloc( new_addr_type addr, mem_access_type type, unsigned size, bool wr ) const = 0;
    virtual mem_fetch *alloc( const class warp_inst_t &inst, const mem_access_t &access ) const = 0;
};

// the maximum number of destination, source, or address uarch operands in a instruction
#define MAX_REG_OPERANDS 8

struct dram_callback_t {
   dram_callback_t() { function=NULL; instruction=NULL; thread=NULL; }
   void (*function)(const class inst_t*, class ptx_thread_info*);

   const class inst_t* instruction;
   class ptx_thread_info *thread;
};


class Histogram {

private:

    std::vector<unsigned> histogramCounterVector;

    std::vector<unsigned> histogramVector;  // 0
    std::vector<double> histogramVector_double; // 1
    std::vector<unsigned long long> histogramVector_long; // 2
    std::vector<std::string> histogramVector_string; // 3

    std::string histogramName;

    unsigned h_sm_id;
    unsigned h_warp_id;

    unsigned type;


public:

    /*Histogram(std::string inputName, unsigned input_sm_id , unsigned input_warp_id) {

        this->histogramName = inputName;
        this->h_sm_id = input_sm_id;
        this->h_warp_id = input_warp_id;

        this->reset_histogram(false);

    }*/


    Histogram(std::string inputName, unsigned input_sm_id , unsigned input_warp_id, unsigned type) {

        this->histogramName = inputName;
        this->h_sm_id = input_sm_id;
        this->h_warp_id = input_warp_id;
        this->type = type;

        this->reset_histogram();

    }


    void reset_histogram () {

        histogramCounterVector.clear();

        histogramVector.clear();
        histogramVector_double.clear();
        histogramVector_long.clear();
        histogramVector_string.clear();

    }

    void update_histogram_double(double input_date){

        assert(type==1);

        bool matched = false;

        for (unsigned i=0; i<histogramVector_double.size(); i++) {

            assert(matched==false);

            /*double epsilon = 0.00000000001;

            if (fabs(histogramVector_double.at(i)-input_date<epsilon)<epsilon){
                histogramCounterVector.at(i)++;
                matched = true;
                break;
            }*/

            if (histogramVector_double.at(i)==input_date){
                histogramCounterVector.at(i)++;
                matched = true;
                break;
            }

        }

        if (!matched) {
            assert(matched==false);
            histogramVector_double.push_back(input_date);
            histogramCounterVector.push_back(1);
        }


    }

    void update_histogram_string(std::string input_date){

        assert(type==3);

        bool matched = false;

        for (unsigned i=0; i<histogramVector_string.size(); i++) {

            assert(matched==false);

            if (histogramVector_string.at(i)==input_date){
                histogramCounterVector.at(i)++;
                matched = true;
                break;
            }

        }

        if (!matched) {
            assert(matched==false);
            histogramVector_string.push_back(input_date);
            histogramCounterVector.push_back(1);
        }

    }


    void update_histogram (unsigned input_data) {


        assert(type==0);

        bool matched = false;

        for (unsigned i=0; i<histogramVector.size(); i++) {

            assert(matched==false);

            if (histogramVector.at(i)==input_data) {
                histogramCounterVector.at(i)++;
                matched = true;
                break;
            }

        }

        if (!matched) {

            assert(matched==false);

            histogramVector.push_back(input_data);
            histogramCounterVector.push_back(1);
        }

    }



    void sort_histogram_byCounter(){

        if (type == 0){
            assert(histogramCounterVector.size() == histogramVector.size());
        }

        if (type == 1){
            assert(histogramCounterVector.size() == histogramVector_double.size());
        }

        if (histogramVector.size()>0){
            for (unsigned i=0; i<histogramCounterVector.size(); i++){
                for (unsigned j=0; j<histogramCounterVector.size()-1; j++){
                    if (histogramCounterVector.at(j)>histogramCounterVector.at(j+1)){
                        unsigned temp = histogramCounterVector.at(j);
                        histogramCounterVector.at(j) = histogramCounterVector.at(j+1);
                        histogramCounterVector.at(j+1) = temp;

                        temp = histogramVector.at(j);
                        histogramVector.at(j)=histogramVector.at(j+1);
                        histogramVector.at(j+1)=temp;
                    }
                }
            }

        }
    }

    void sort_histogram_byName(){

        if (type == 0){

            assert(histogramCounterVector.size() == histogramVector.size());

            if (histogramVector.size()>0){

                for (unsigned i=0; i<histogramVector.size(); i++){
                    for (unsigned j=0; j<histogramVector.size()-1; j++){
                        if (histogramVector.at(j)>histogramVector.at(j+1)){

                            unsigned temp = histogramCounterVector.at(j);
                            histogramCounterVector.at(j) = histogramCounterVector.at(j+1);
                            histogramCounterVector.at(j+1) = temp;

                            temp = histogramVector.at(j);
                            histogramVector.at(j)=histogramVector.at(j+1);
                            histogramVector.at(j+1)=temp;

                        }
                    }
                }

            }
        }

        if (type == 1) {

            assert(histogramCounterVector.size() == histogramVector_double.size());

            if (histogramVector_double.size() > 0) {
                for (unsigned i = 0; i < histogramVector_double.size(); i++) {
                    for (unsigned j = 0; j < histogramVector_double.size() - 1; j++) {
                        if (histogramVector_double.at(j) > histogramVector_double.at(j + 1)) {

                            unsigned temp = histogramCounterVector.at(j);
                            histogramCounterVector.at(j) = histogramCounterVector.at(j + 1);
                            histogramCounterVector.at(j + 1) = temp;

                            double tempD;
                            tempD = histogramVector_double.at(j);
                            histogramVector_double.at(j) = histogramVector_double.at(j + 1);
                            histogramVector_double.at(j + 1) = tempD;

                        }
                    }
                }
            }
        }

        if (type == 3) {

            assert(histogramCounterVector.size() == histogramVector_string.size());

            if (histogramVector_string.size() > 0) {
                for (unsigned i = 0; i < histogramVector_string.size(); i++) {
                    for (unsigned j = 0; j < histogramVector_string.size() - 1; j++) {
                        if (histogramVector_string.at(j) > histogramVector_string.at(j + 1)) {

                            unsigned temp = histogramCounterVector.at(j);
                            histogramCounterVector.at(j) = histogramCounterVector.at(j + 1);
                            histogramCounterVector.at(j + 1) = temp;

                            std::string tempS;
                            tempS = histogramVector_string.at(j);
                            histogramVector_string.at(j) = histogramVector_string.at(j + 1);
                            histogramVector_string.at(j + 1) = tempS;

                        }
                    }
                }
            }
        }

    }

    bool is_empty(){
        if (histogramVector.size()>0){
            return true;
        }
        else {
            return false;
        }
    }

    void print_histogram() {

        if (type == 0){

            assert(histogramCounterVector.size() == histogramVector.size());

            this->sort_histogram_byName();

            if (histogramVector.size()>0) {

                char headerBuffer[100];

                unsigned headerSize = sprintf(headerBuffer,
                                              "######################### DRSVR - [SM: %u ; WARP: %u] - %s #########################\n",
                                              this->h_sm_id, this->h_warp_id, histogramName.c_str());

                printf("######################### DRSVR - [SM: %u ; WARP: %u] - %s #########################\n"
                        , this->h_sm_id
                        , this->h_warp_id
                        , histogramName.c_str());

                for (unsigned i=0; i<histogramVector.size(); i++){
                    printf("%s[%u] : %u\n"
                            , histogramName.c_str()
                            , histogramVector.at(i)
                            , histogramCounterVector.at(i));
                }

                for (unsigned i=0; i<headerSize-1; i++){
                    printf("#");
                }

                printf("\n\n");

            }

        }

        if (type == 1){

            assert(histogramCounterVector.size() == histogramVector_double.size());

            this->sort_histogram_byName();

            if (histogramVector_double.size()>0) {

                char headerBuffer[100];

                unsigned headerSize = sprintf(headerBuffer,
                                              "######################### DRSVR - [SM: %u ; WARP: %u] - %s #########################\n",
                                              this->h_sm_id, this->h_warp_id, histogramName.c_str());

                printf("######################### DRSVR - [SM: %u ; WARP: %u] - %s #########################\n"
                        , this->h_sm_id
                        , this->h_warp_id
                        , histogramName.c_str());

                for (unsigned i=0; i<histogramVector_double.size(); i++){
                    printf("%s[%4.4f] : %u\n"
                            , histogramName.c_str()
                            , histogramVector_double.at(i)
                            , histogramCounterVector.at(i));
                }

                for (unsigned i=0; i<headerSize-1; i++){
                    printf("#");
                }

                printf("\n\n");

            }


        }

        if (type == 3){

            assert(histogramCounterVector.size() == histogramVector_string.size());

            this->sort_histogram_byName();

            if (histogramVector_string.size()>0) {

                char headerBuffer[100];

                unsigned headerSize = sprintf(headerBuffer,
                                              "######################### DRSVR - [SM: %u ; WARP: %u] - %s #########################\n",
                                              this->h_sm_id, this->h_warp_id, histogramName.c_str());

                printf("######################### DRSVR - [SM: %u ; WARP: %u] - %s #########################\n"
                        , this->h_sm_id
                        , this->h_warp_id
                        , histogramName.c_str());

                for (unsigned i=0; i<histogramVector_string.size(); i++){
                    printf("%s[%s] : %u\n"
                            , histogramName.c_str()
                            , histogramVector_string.at(i).c_str()
                            , histogramCounterVector.at(i));
                }

                for (unsigned i=0; i<headerSize-1; i++){
                    printf("#");
                }

                printf("\n\n");

            }


        }


    }
};

struct dlcEntry {
    unsigned PC;
    unsigned instCounter;
    unsigned accCounter;
    unsigned setCounter;
    unsigned missCounter;


    dlcEntry(unsigned input_PC, unsigned accCount, unsigned setCount){
        PC = input_PC;
        instCounter = 1;
        accCounter = accCount;
        setCounter = setCount;
    }


    dlcEntry(unsigned input_PC, unsigned instCount, unsigned accCount, unsigned setCount, unsigned missCount){
        PC = input_PC;
        instCounter = instCount;
        accCounter = accCount;
        setCounter = setCount;
        missCounter = missCount;
    }

};


class DLC {

private:

    unsigned numberOfSets;
    unsigned numberOfSetsLog2;
    unsigned wordSize;
    unsigned numberOfOffsetLog2;
    unsigned setMask;

    unsigned tempPC;

    std::map<unsigned,dlcEntry*> dlcTable;

    std::vector<unsigned> transaction_history_vector;


    void update_dlc_entry(unsigned input_PC, unsigned input_inst, unsigned input_acc, unsigned input_set, unsigned input_miss){
        dlcTable.at(input_PC)->PC = input_PC;
        dlcTable.at(input_PC)->instCounter+=input_inst;
        dlcTable.at(input_PC)->accCounter+=input_acc;
        dlcTable.at(input_PC)->setCounter+=input_set;
        if (dlcTable.at(input_PC)->missCounter < input_miss){
            dlcTable.at(input_PC)->missCounter = input_miss;
        }
    }

    void update_inst(unsigned input_PC) {
        dlcTable.at(input_PC)->instCounter++;
    }

    void update_miss(unsigned input_PC, unsigned input_miss){
        if (dlcTable.at(input_PC)->missCounter < input_miss){
            dlcTable.at(input_PC)->missCounter = input_miss;
        }

    }

    void update_acc (unsigned input_PC, unsigned transaction_count){
        dlcTable.at(input_PC)->accCounter+=transaction_count;
    }

    void update_set (unsigned input_PC, unsigned set_count){
        dlcTable.at(input_PC)->setCounter+=set_count;
    }

    unsigned get_inst(unsigned input_PC) {
        return (dlcTable.at(input_PC)->instCounter);
    }

    unsigned get_acc (unsigned input_PC){
        return(dlcTable.at(input_PC)->accCounter);
    }

    unsigned get_set (unsigned input_PC){
        return (dlcTable.at(input_PC)->setCounter);
    }

    unsigned get_missCount (unsigned input_PC){
        return (dlcTable.at(input_PC)->missCounter);
    }


    bool findPC (unsigned input_pc) {

        std::map<unsigned,dlcEntry*>::iterator it = dlcTable.find(input_pc);

        if (it != dlcTable.end()){
            return  true;
        }
        else {
            return false;
        }

    }

    void update_transaction_history (std::vector<unsigned> transaction_vector_in){
        for (unsigned i=0; i<transaction_vector_in.size(); i++){
            transaction_history_vector.push_back(transaction_vector_in.at(i));
        }
    }

    void initialize_setMask(){

        //printf("Initialize setMask\n");

        unsigned offsetBits = 0;
        unsigned tempWordSize = wordSize;

        while (true){
            tempWordSize = tempWordSize / 2 ;
            if (tempWordSize == 0){
                break;
            }
            offsetBits++;
        }

        numberOfOffsetLog2 = offsetBits;

        unsigned setBits = numberOfSetsLog2;  // Number of bits for indexing sets inside the address

        unsigned boundBits = setBits+offsetBits; // Offsetbit + SetBit

        unsigned bitIndex = 0;

        setMask = 0;

        unsigned pow2 = 1;

        while (true){

            if (bitIndex>=offsetBits){
                if (bitIndex<boundBits){
                    setMask = setMask + pow2;
                    //printf("DRSVR setMask: %u;\n",setMask);
                    //printf("setMask:%u ; pow2:%u ; \n", setMask, pow2);
                }
                else {
                    break;
                }

            }

            pow2 = pow2 * 2;
            bitIndex++;

        }

    }

    unsigned calculateSet (unsigned block_address){

        unsigned set = block_address&(setMask);

        set = set >> numberOfOffsetLog2;

        //printf("Block_Address:%u ; setMask:%u ; set:%u \n", block_address, setMask, set);

        assert(set<32);

        return set;


    }

    unsigned getSetsCount (std::vector<unsigned> input_Vector){

        unsigned setCount = 0;

        std::bitset<32> sets_Bitset;


        for (unsigned i=0; i<input_Vector.size(); i++){
            unsigned testSet = calculateSet(input_Vector.at(i));

            sets_Bitset.set(testSet);

            //printf("DRSVR-DLC PC:%u ; Address:%u ; testSet:%u ; bitSet:%s setCount:%u ;\n"
            //        ,tempPC , input_Vector.at(i), testSet, sets_Bitset.to_string().c_str(), setCount );

        }

        if (sets_Bitset.size()>0){
            setCount = sets_Bitset.count();
        }

        //printf("DRSVR-DLC bitSet:%s setCount:%u ;\n"
        //        , sets_Bitset.to_string().c_str(), setCount );

        return  setCount;


    }


    //template <class T>
    unsigned numDigits(unsigned number)
    {
        unsigned digits = 0;
        if (number < 0) digits = 1; // remove this line if '-' counts as a digit
        while (number) {
            number /= 10;
            digits++;
        }
        return digits;
    }


public:

    DLC(unsigned cache_set_number, unsigned cache_set_number_log2, unsigned word_size_in) {
        numberOfSets = cache_set_number;
        numberOfSetsLog2 = cache_set_number_log2;
        wordSize = word_size_in;
        initialize_setMask();
        printf("DLC HAS BEEN CREATED!\t #Sets: %u ; #setBits: %u ; wordSize: %u; setMask: %u\n"
                , numberOfSets, numberOfSetsLog2, wordSize, setMask);
    }


    void aggragateDLC (DLC* input_dlc){
        for (std::map<unsigned,dlcEntry*>::iterator it = dlcTable.begin(); it!=dlcTable.end(); ++it){

            unsigned input_pc = it->second->PC;
            unsigned input_inst = it->second->instCounter;
            unsigned input_acc = it->second->accCounter;
            unsigned input_set = it->second->setCounter;
            unsigned input_miss = it->second->missCounter;

            input_dlc->backup_DLC_entry(input_pc,input_inst,input_acc,input_set,input_miss);
        }
    }

    /*void update_DLC (unsigned input_pc, std::vector<unsigned> transaction_vector ){

        tempPC = input_pc;

        update_transaction_history(transaction_vector);

        unsigned transaction_count = transaction_vector.size();
        bool pcFound = findPC(input_pc);

        if (pcFound){
            update_inst(input_pc);
            update_acc(input_pc,transaction_count);
            update_set(input_pc,getSetsCount(transaction_vector));
        }
        else {
            dlcEntry *test = new dlcEntry(input_pc, transaction_count, getSetsCount(transaction_vector));
            dlcTable[input_pc] = test;
        }
    }*/

    void update_DLC_entry (unsigned input_pc, unsigned input_inst, unsigned input_acc, unsigned input_set, unsigned input_miss){

        tempPC = input_pc;

        bool pcFound = findPC(input_pc);

        if (pcFound){
            update_dlc_entry(input_pc, input_inst, input_acc, input_set, input_miss);
        }
        else {
            dlcEntry *test = new dlcEntry(input_pc, input_inst, input_acc, input_set, input_miss);
            dlcTable[input_pc] = test;
        }
    }

    void backup_DLC_entry(unsigned input_pc, unsigned input_inst, unsigned input_acc, unsigned input_set, unsigned input_miss){

        tempPC = input_pc;

        bool pcFound = findPC(input_pc);

        if (pcFound){
            update_dlc_entry(input_pc, input_inst, input_acc, input_set, input_miss);
        }
        else {
            dlcEntry *test = new dlcEntry(input_pc, input_inst, input_acc, input_set, input_miss);
            dlcTable[input_pc] = test;
        }
    }


    void printTofile_DLC(FILE *dlcFile, unsigned sm_id){

        fprintf(dlcFile, "-------------------------------------------------------------------------------------------------------------\n");
        fprintf(dlcFile, "                                             DLC TABLE [%u]\n",sm_id);
        fprintf(dlcFile, "-------------------------------------------------------------------------------------------------------------\n");
        fprintf(dlcFile, "|           PC           |           #INST           |           #ACC           |           #SETS           |\n");
        fprintf(dlcFile, "-------------------------------------------------------------------------------------------------------------\n");

        for (std::map<unsigned,dlcEntry*>::iterator it = dlcTable.begin(); it!=dlcTable.end(); ++it){

            // Print PC
            unsigned numDigit = this->numDigits(it->second->PC);
            unsigned totalSpace = 2 + 11 ;
            unsigned whiteSpaces = totalSpace - numDigit;

            fprintf(dlcFile, "|           %u",it->second->PC);
            for (unsigned i=0; i<whiteSpaces; i++){
                fprintf(dlcFile, " ");
            }


            // Print INST
            numDigit = this->numDigits(it->second->instCounter);
            totalSpace = 5 + 11 ;
            whiteSpaces = totalSpace - numDigit;

            fprintf(dlcFile, "|           %u",it->second->instCounter);
            for (unsigned i=0; i<whiteSpaces; i++){
                fprintf(dlcFile, " ");
            }

            numDigit = this->numDigits(it->second->accCounter);
            totalSpace = 4 + 11 ;
            whiteSpaces = totalSpace - numDigit;

            fprintf(dlcFile, "|           %u",it->second->accCounter);
            for (unsigned i=0; i<whiteSpaces; i++){
                fprintf(dlcFile, " ");
            }

            numDigit = this->numDigits(it->second->setCounter);
            totalSpace = 5 + 11 ;
            whiteSpaces = totalSpace - numDigit;

            fprintf(dlcFile, "|           %u",it->second->setCounter);
            for (unsigned i=0; i<whiteSpaces; i++){
                fprintf(dlcFile, " ");
            }

            fprintf(dlcFile, "|\n");

            /*printf("|           %u           |           %u           |           %u           |           %u           |\n"
            ,it->first, it->second->instCounter, it->second->accCounter, it->second->setCounter);*/
        }

        fprintf(dlcFile, "-------------------------------------------------------------------------------------------------------------\n");

    }

    void print_DLC(unsigned sm_id){

        printf("-----------------------------------------------------------------------------------------------------------------------------------------\n");
        printf("                                             DLC TABLE [%u]\n",sm_id);
        printf("-----------------------------------------------------------------------------------------------------------------------------------------\n");
        printf("|           PC           |           #INST           |           #ACC           |           #SETS           |           #MISS           |\n");
        printf("-----------------------------------------------------------------------------------------------------------------------------------------\n");

        for (std::map<unsigned,dlcEntry*>::iterator it = dlcTable.begin(); it!=dlcTable.end(); ++it){

            // Print PC
            unsigned numDigit = this->numDigits(it->second->PC);
            unsigned totalSpace = 2 + 11 ;
            unsigned whiteSpaces = totalSpace - numDigit;

            printf("|           %u",it->second->PC);
            for (unsigned i=0; i<whiteSpaces; i++){
                printf(" ");
            }


            // Print INST
            numDigit = this->numDigits(it->second->instCounter);
            totalSpace = 5 + 11 ;
            whiteSpaces = totalSpace - numDigit;

            printf("|           %u",it->second->instCounter);
            for (unsigned i=0; i<whiteSpaces; i++){
                printf(" ");
            }

            numDigit = this->numDigits(it->second->accCounter);
            totalSpace = 4 + 11 ;
            whiteSpaces = totalSpace - numDigit;

            printf("|           %u",it->second->accCounter);
            for (unsigned i=0; i<whiteSpaces; i++){
                printf(" ");
            }

            numDigit = this->numDigits(it->second->setCounter);
            totalSpace = 5 + 11 ;
            whiteSpaces = totalSpace - numDigit;

            printf("|           %u",it->second->setCounter);
            for (unsigned i=0; i<whiteSpaces; i++){
                printf(" ");
            }

            numDigit = this->numDigits(it->second->missCounter);
            totalSpace = 5 + 11 ;
            whiteSpaces = totalSpace - numDigit;

            printf("|           %u",it->second->missCounter);
            for (unsigned i=0; i<whiteSpaces; i++){
                printf(" ");
            }

            printf("|\n");

            /*printf("|           %u           |           %u           |           %u           |           %u           |\n"
            ,it->first, it->second->instCounter, it->second->accCounter, it->second->setCounter);*/
        }

        printf("-----------------------------------------------------------------------------------------------------------------------------------------\n");

    }



    void print_DLC0(unsigned sm_id){

        printf("-------------------------------------------------------------------------------------------------------------\n");
        printf("                                             DLC TABLE [%u]\n",sm_id);
        printf("-------------------------------------------------------------------------------------------------------------\n");
        printf("|           PC           |           #INST           |           #ACC           |           #SETS           |\n");
        printf("-------------------------------------------------------------------------------------------------------------\n");

        for (std::map<unsigned,dlcEntry*>::iterator it = dlcTable.begin(); it!=dlcTable.end(); ++it){

            // Print PC
            unsigned numDigit = this->numDigits(it->second->PC);
            unsigned totalSpace = 2 + 11 ;
            unsigned whiteSpaces = totalSpace - numDigit;

            printf("|           %u",it->second->PC);
            for (unsigned i=0; i<whiteSpaces; i++){
                printf(" ");
            }


            // Print INST
            numDigit = this->numDigits(it->second->instCounter);
            totalSpace = 5 + 11 ;
            whiteSpaces = totalSpace - numDigit;

            printf("|           %u",it->second->instCounter);
            for (unsigned i=0; i<whiteSpaces; i++){
                printf(" ");
            }

            numDigit = this->numDigits(it->second->accCounter);
            totalSpace = 4 + 11 ;
            whiteSpaces = totalSpace - numDigit;

            printf("|           %u",it->second->accCounter);
            for (unsigned i=0; i<whiteSpaces; i++){
                printf(" ");
            }

            numDigit = this->numDigits(it->second->setCounter);
            totalSpace = 5 + 11 ;
            whiteSpaces = totalSpace - numDigit;

            printf("|           %u",it->second->setCounter);
            for (unsigned i=0; i<whiteSpaces; i++){
                printf(" ");
            }

            printf("|\n");

            /*printf("|           %u           |           %u           |           %u           |           %u           |\n"
            ,it->first, it->second->instCounter, it->second->accCounter, it->second->setCounter);*/
        }

        printf("-------------------------------------------------------------------------------------------------------------\n");

    }


    void print_transactions_history(){
        unsigned setCount_Sum = 0;
        printf("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n");
        printf("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$DRSVR DLC TRANSACTION HISTORY$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n");
        printf("-----------------------------------------------------------------------------------------\n");
        printf("wordSize: %u ; OffsetBits: %u ; setBits: %u; setMask: %u;\n",wordSize, numberOfOffsetLog2, numberOfSetsLog2, setMask );
        printf("-----------------------------------------------------------------------------------------\n");
        for (unsigned i=0; i<transaction_history_vector.size(); i++ ) {
            setCount_Sum += calculateSet(transaction_history_vector.at(i));
            printf("Transaction: %u ; Set: %u  ;\n",transaction_history_vector.at(i), calculateSet(transaction_history_vector.at(i)));
        }
        printf("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n");
    }

    bool isDivergent (unsigned input_pc){
        return (findPC(input_pc));
    }

    unsigned get_SetTouched (unsigned input_pc){
        return (get_set(input_pc));
    }

    unsigned get_InstOccurance (unsigned input_pc){
        return (get_inst(input_pc));
    }

    unsigned get_TransactionCounts (unsigned input_pc){
        return (get_acc(input_pc));
    }

    unsigned get_MissCount (unsigned input_pc){
        return (get_missCount(input_pc));
    }

};


class OCW_LOGIC{


private:

    bool Div;
    bool FCL;


    unsigned Set;
    unsigned Acc;

    unsigned OCW;


    unsigned CNT;
    unsigned Delta;

    const unsigned  CMAX = 255;
    const unsigned  WMAX = 48 ;


    void calculateDelta(bool debugMode){

        double tempSet = static_cast<double>(Set);
        double tempAcc = static_cast<double>(Acc);

        if (Acc>Set*1.5){
            if (debugMode){
                printf("Set:%d ; Acc:%d Acc/Set:%f\n", Set, Acc, tempAcc/tempSet);
                printf("DRSVR DELTA CHANGE!\n");
            }
            if (CNT>2){
                Delta = CNT/2;
            }
            // Associativity Sensitive
        }
        else{
            Delta = 1;
        }
    }

    void OCW_Estimation_Logic (bool debugMode){


        if (Div){
            calculateDelta(debugMode);

            // Should be a divergent Load


            if (debugMode){
                printf("DRSVR OCW LOGIC BEGIN: FCL:%u ; CNT:%u ; Delta:%u; OCW:%u ;\n",FCL, CNT, Delta, OCW);
            }

            if (FCL){
                CNT++;
                if ( (CNT>=CMAX) && (OCW<WMAX)){
                    OCW = OCW++;
                    CNT=0;
                }
            }
            else {
                if (CNT > Delta) {
                    CNT -= Delta;
                } else {
                    CNT = 255;
                    if (OCW > 2) {
                        OCW--;
                    }
                }
            }

            if (debugMode){
                printf("DRSVR OCW LOGIC END: FCL:%u ; CNT:%u ; Delta:%u; OCW:%u ;\n",FCL, CNT, Delta, OCW);
            }
        }

    }


public:

    OCW_LOGIC(){
        Div = false;
        FCL = false;
        CNT = 127;
        Acc = 0;
        OCW = 2;
    }

    void isLoadDivergent(bool isDivergent){
        Div = isDivergent;
    }

    bool getDiv (){
        return Div;
    }

    void isLoadFullyCached(bool isFullyCached){
        FCL = isFullyCached;
    }

    void updateOCW_SET_ACC(unsigned set_in, unsigned acc_in){
        Acc = acc_in;
        Set = set_in;
    }

    bool getFCL (){
        return FCL;
    }

    /*void setInfoSetAndAcc (unsigned in_Set, unsigned in_Acc){
        Set = in_Set;
        Acc = in_Acc;
    }*/

    /*void getNumberOfTouchedSets (){
        return Set;
    }*/

    unsigned getOCW(bool debugMode){
        bool debug = debugMode;
        OCW_Estimation_Logic(debug);
        return OCW;
    }


};

class DRSVRSTATS {

private:

    std::vector<std::string> stats_name_vector;
    std::vector<Histogram*> stats_obj_vector;

    std::map< std::string, std::map < unsigned, Histogram*, std::less<unsigned> > > stats_obj_PCMAP;


    unsigned  s_sm_id;
    unsigned  s_warp_id;

    Histogram* create_new_histogram_obj (std::string myHistogramName, unsigned type) {
        Histogram *warp_parts_obj = new Histogram(myHistogramName, s_sm_id, s_warp_id, type);
        return (warp_parts_obj);
    }


    unsigned get_last_element_index() {
        if (this->stats_name_vector.size()>0){
            return (this->stats_name_vector.size()-1);
        }
    }

    unsigned get_stat_vector_size() {
        if (stats_name_vector.size()!=stats_obj_vector.size()){
            stats_obj_vector.clear();
            stats_name_vector.clear();
            printf("Something is Wrong: Name:%u ; Obj:%u\n SM_ID:%u ; WARP_ID:%u ;", stats_name_vector.size(), stats_obj_vector.size(), s_sm_id, s_warp_id);
        }
        assert(stats_name_vector.size()==stats_obj_vector.size());
        return (this->stats_name_vector.size());
    }

    unsigned search_histogram(std::string myStatName) {

        unsigned index = this->get_stat_vector_size();

        for (unsigned i=0; i<stats_name_vector.size(); i++){
            if (stats_name_vector.at(i)==myStatName){
                index = i;
                break;
            }
        }
        return index ;
    }

    bool find_histogram(std::string myStatName){

        unsigned statVectorSize = this->get_stat_vector_size();
        unsigned foundIndex = this->search_histogram(myStatName);

        if (foundIndex == statVectorSize){
            return false;
        }
        else {
            return true;
        }

    }

public:

    DRSVRSTATS (unsigned sm_id_input, unsigned warp_id_input) {
        s_sm_id = sm_id_input;
        s_warp_id = warp_id_input;
        stats_name_vector.clear();
        stats_obj_vector.clear();
        stats_obj_PCMAP.clear();
    }

    void update_histogram(std::string histogramName_input, unsigned input_data){

        unsigned statVectorSize = get_stat_vector_size();
        unsigned foundIndex = search_histogram(histogramName_input);

        if (foundIndex == statVectorSize){
            // Create a new Histogram
            stats_name_vector.push_back(histogramName_input);
            stats_obj_vector.push_back(create_new_histogram_obj(histogramName_input,0));
            foundIndex = get_last_element_index();
        }

        // Update an existing histogram
        if (stats_obj_vector.size()>0){
            assert(foundIndex<stats_obj_vector.size());
            this->stats_obj_vector.at(foundIndex)->update_histogram(input_data);
        }

    }


    void update_histogram_double(std::string histogramName_input, double input_data){

        unsigned statVectorSize = get_stat_vector_size();
        unsigned foundIndex = search_histogram(histogramName_input);

        if (foundIndex == statVectorSize){
            // Create a new Histogram
            stats_name_vector.push_back(histogramName_input);
            stats_obj_vector.push_back(create_new_histogram_obj(histogramName_input,1));
            foundIndex = get_last_element_index();
        }


        // Update an existing histogram
        if (stats_obj_vector.size()>0){
            assert(foundIndex<stats_obj_vector.size());
            this->stats_obj_vector.at(foundIndex)->update_histogram_double(input_data);
        }

    }



    void update_histogram_string(std::string histogramName_input, std::string input_data){

        unsigned statVectorSize = get_stat_vector_size();
        unsigned foundIndex = search_histogram(histogramName_input);

        if (foundIndex == statVectorSize){
            // Create a new Histogram
            stats_name_vector.push_back(histogramName_input);
            stats_obj_vector.push_back(create_new_histogram_obj(histogramName_input,3));
            foundIndex = get_last_element_index();
        }


        // Update an existing histogram
        if (stats_obj_vector.size()>0){
            assert(foundIndex<stats_obj_vector.size());
            this->stats_obj_vector.at(foundIndex)->update_histogram_string(input_data);
        }

    }




    void print_histogram (std::string histogramName_input){

        unsigned statVectorSize = this->get_stat_vector_size();
        unsigned foundIndex = this->search_histogram(histogramName_input);

        if (foundIndex != statVectorSize){
            this->stats_obj_vector.at(foundIndex)->print_histogram();
        }

    }

    void reset_histogram (std::string histogramName_input){

        unsigned statVectorSize = this->get_stat_vector_size();
        unsigned foundIndex = this->search_histogram(histogramName_input);

        if (foundIndex != statVectorSize){
            this->stats_obj_vector.at(foundIndex)->reset_histogram();
        }

    }

    void update_PC_Histogram(std::string input_name , unsigned input_pc,  unsigned input_date) {

        std::map< std::string, std::map < unsigned, Histogram*, std::less<unsigned> > > :: iterator it_name;

        it_name = stats_obj_PCMAP.find(input_name);

        if (it_name == stats_obj_PCMAP.end()){

            std::map< unsigned, Histogram*, std::less<unsigned> > stats_obj_PCMAP_obj ;

            stats_obj_PCMAP_obj[input_pc] = create_new_histogram_obj(input_name, 0);

            stats_obj_PCMAP[input_name] = stats_obj_PCMAP_obj;

        }

        else {

            std::map< unsigned, Histogram*, std::less<unsigned> >::iterator it_pc;

            it_pc = stats_obj_PCMAP.at(input_name).find(input_pc);

            if (it_pc == stats_obj_PCMAP.at(input_name).end()){

                stats_obj_PCMAP.at(input_name)[input_pc] = create_new_histogram_obj(input_name, 0);
            }
        }

        // UPDATE

        stats_obj_PCMAP.at(input_name).at(input_pc)->update_histogram(input_date);

    }

    void update_PC_Histogram_double(std::string input_name , unsigned input_pc, double input_date) {

        std::map< std::string, std::map < unsigned, Histogram*, std::less<unsigned> > > :: iterator it_name;

        it_name = stats_obj_PCMAP.find(input_name);

        if (it_name == stats_obj_PCMAP.end()){

            std::map< unsigned, Histogram*, std::less<unsigned> > stats_obj_PCMAP_obj ;

            stats_obj_PCMAP_obj[input_pc] = create_new_histogram_obj(input_name, 1);

            stats_obj_PCMAP[input_name] = stats_obj_PCMAP_obj;

        }

        else {

            std::map< unsigned, Histogram*, std::less<unsigned> >::iterator it_pc;

            it_pc = stats_obj_PCMAP.at(input_name).find(input_pc);

            if (it_pc == stats_obj_PCMAP.at(input_name).end()){

                stats_obj_PCMAP.at(input_name)[input_pc] = create_new_histogram_obj(input_name, 1);
            }
        }

        // UPDATE

        stats_obj_PCMAP.at(input_name).at(input_pc)->update_histogram_double(input_date);

    }

    void update_PC_Histogram_string(std::string input_name , unsigned input_pc, std::string input_date) {

        std::map< std::string, std::map < unsigned, Histogram*, std::less<unsigned> > > :: iterator it_name;

        it_name = stats_obj_PCMAP.find(input_name);

        if (it_name == stats_obj_PCMAP.end()){

            std::map< unsigned, Histogram*, std::less<unsigned> > stats_obj_PCMAP_obj ;

            stats_obj_PCMAP_obj[input_pc] = create_new_histogram_obj(input_name, 3);

            stats_obj_PCMAP[input_name] = stats_obj_PCMAP_obj;

        }

        else {

            std::map< unsigned, Histogram*, std::less<unsigned> >::iterator it_pc;

            it_pc = stats_obj_PCMAP.at(input_name).find(input_pc);

            if (it_pc == stats_obj_PCMAP.at(input_name).end()){

                stats_obj_PCMAP.at(input_name)[input_pc] = create_new_histogram_obj(input_name, 3);
            }
        }

        // UPDATE

        stats_obj_PCMAP.at(input_name).at(input_pc)->update_histogram_string(input_date);

    }





    void print_PC_Histogram(std::string input_name){

        std::map< std::string, std::map < unsigned, Histogram*, std::less<unsigned> > > :: iterator it_name;

        it_name = stats_obj_PCMAP.find(input_name);

        if (it_name != stats_obj_PCMAP.end()){

            std::map< unsigned, Histogram*, std::less<unsigned> >::iterator it_pc;

            for (it_pc = stats_obj_PCMAP.at(input_name).begin(); it_pc != stats_obj_PCMAP.at(input_name).end(); ++it_pc ){


                printf("%sPC[%u]:\n", input_name.c_str(), it_pc->first );

                it_pc->second->print_histogram();

                printf("\n");

            }

        }

    }


};


class FCLUnit {

private:

    bool debugMode = false;

    bool startTracking;
    bool dlcLock;

    unsigned PC;
    unsigned WARPID;
    unsigned SMID;

    unsigned LOAD_COUNT;
    unsigned MISS_COUNT;
    unsigned HIT_COUNT;

    double ratio;


    std::bitset<64> SetBitSet;
    unsigned set;
    unsigned acc;
    unsigned inst;

    bool FCL;
    bool FCLValid;


    unsigned Counter;

    void resetStat() {

        //debugMode = false;

        startTracking = false;
        dlcLock = false;


        PC = (unsigned)-1;
        WARPID = (unsigned)-1;
        SMID = (unsigned)-1;

        LOAD_COUNT = 0;
        MISS_COUNT = 0;
        HIT_COUNT = 0;

        Counter = 0;

        resetSet();
        set = 0;
        acc = 0;
        inst = 0;
        FCL = false;
        FCLValid = false;
    }

    bool analiyzeStatus(unsigned status){
        switch (status) {
            case 0:{
                if (debugMode){
                    printf("DRSVR HIT STATUS!\n");
                }
                return true;
                break;
            }
            case 1:{
                if (debugMode){
                    printf("DRSVR HIT_RESERVED STATUS!\n");
                }
                return false;
                break;
            }
            case 2:{
                if (debugMode){
                    printf("DRSVR MISS STATUS!\n");
                }
                return false;
                break;
            }
            case 3:{
                if (debugMode){
                    printf("DRSVR RESERVATION FAIL STATUS!\n");
                }
                return false;
                break;
            }
        }
    }

    void updateStatus(unsigned status_in, unsigned WARP_ID_IN, unsigned SM_ID_IN, unsigned PC_IN, unsigned SET_IN){
        //debugMode = false ;

        this->checkInfo(WARP_ID_IN, SM_ID_IN, PC_IN);
        bool status = this->analiyzeStatus(status_in);
        Counter++;
        if (status == 0){
            MISS_COUNT++;
        }
        else {
            HIT_COUNT++;
        }
        updateSetBitset(SET_IN);
    }

    void checkInfo(unsigned WARP_ID_IN, unsigned SM_ID_IN, unsigned PC_IN){
        assert(WARP_ID_IN==WARPID);
        assert(SM_ID_IN==SMID);
        assert(PC_IN==PC);
    }

    void updateSetBitset(unsigned set_in){
        SetBitSet.set(set_in);
    }

    void resetSet(){
        SetBitSet.reset();
    }

    unsigned getSetTouched(){
        return SetBitSet.count();
    }

    unsigned getAccCount(){
        return LOAD_COUNT;
    }

    bool lockDLC(){

        //Should be at the end of tracking
        assert(Counter==LOAD_COUNT);

        acc = getAccCount();
        set = getSetTouched();
        inst = 1;

        if (debugMode){
            printf("DRSVR FCL UNIT DLCLOCKED! [%u;%u;%u]\t", WARPID, SMID, PC);
            printf("PC:%u ; INST:%u ; ACC:%u ; SET:%u;\n", PC ,inst ,acc ,set );
        }

        dlcLock = true;

    }





public:

    FCLUnit(){
        this->resetStat();
    }

    void enableDebugMode(){
        debugMode = true;
    }

    unsigned getMissCount(){
        assert(FCLValid);
        return MISS_COUNT;
    }

    unsigned getHitCount(){
        assert(FCLValid);
        return HIT_COUNT;
    }

    unsigned getLoadCount(){
        return LOAD_COUNT;
    }

    double getRatio(){
        assert(FCLValid);
        return ratio;
    }

    void update_FCLUnit(unsigned WARP_ID_IN, unsigned SM_ID_IN, unsigned PC_IN, unsigned LOAD_COUNT_IN, unsigned status, unsigned set, bool is_write){

        bool isLoad = !is_write;
        bool isDivergent = (LOAD_COUNT_IN>1);

        if ( isLoad ){
            if (!startTracking && isDivergent){
                // START TRACKING
                this->resetStat();
                if (debugMode){
                    printf("DRSVR START FCLUNIT TRACKING!\n");
                }
                startTracking = true;
                PC = PC_IN;
                WARPID = WARP_ID_IN;
                SMID = SM_ID_IN;
                LOAD_COUNT = LOAD_COUNT_IN;
            }

            if (startTracking){

                updateStatus(status, WARP_ID_IN, SM_ID_IN, PC_IN, set);

                if (debugMode){
                    this->print();
                }

                if (Counter==LOAD_COUNT){
                    if (debugMode){
                        printf("DRSVR END FCLUNIT TRACKING!\n");
                    }
                    startTracking = false;
                    lockDLC();


                    if (MISS_COUNT==0){
                        FCL= true;
                    }
                    else {
                        FCL = false;
                    }

                    ratio = static_cast<double>(HIT_COUNT)/LOAD_COUNT;

                    FCLValid = true;

                }

            }
        }
    }

    void print(){
        printf("DRSVR FCU UNIT! START:%u; WarpID/SMID/PC:[%u;%u;%u] ; Processed:%u/%u ; MISS_COUNT:%u; HIT_COUNT:%u\n"
                ,startTracking
                ,WARPID ,SMID, PC
                ,Counter, LOAD_COUNT ,MISS_COUNT, HIT_COUNT
        );
    }

    bool isDivergent(){
        return (LOAD_COUNT>1);
    }

    bool isDone(){
        return (dlcLock);
    }

    bool isFCL(){
        assert(this->isDivergent());
        assert(this->isDone());
        assert(this->FCLValid);
        return (FCL);
    }

    unsigned* getDLCEntry(){

        assert(dlcLock);

        unsigned *DLCEntry = new unsigned[4];

        DLCEntry[0] = PC;
        DLCEntry[1] = inst;
        DLCEntry[2] = acc;
        DLCEntry[3] = set;

        return DLCEntry;

    }

};












class inst_t {
public:

    unsigned  DRSVR_COALESCE;

    DRSVR *smObj;

    inst_t()
    {
        m_decoded=false;
        pc=(address_type)-1;
        reconvergence_pc=(address_type)-1;
        op=NO_OP;
        bar_type=NOT_BAR;
        red_type=NOT_RED;
        bar_id=(unsigned)-1;
        bar_count=(unsigned)-1;
        oprnd_type=UN_OP;
        sp_op=OTHER_OP;
        op_pipe=UNKOWN_OP;
        mem_op=NOT_TEX;
        num_operands=0;
        num_regs=0;
        memset(out, 0, sizeof(unsigned));
        memset(in, 0, sizeof(unsigned));
        is_vectorin=0;
        is_vectorout=0;
        space = memory_space_t();
        cache_op = CACHE_UNDEFINED;
        latency = 1;
        initiation_interval = 1;
        for( unsigned i=0; i < MAX_REG_OPERANDS; i++ ) {
            arch_reg.src[i] = -1;
            arch_reg.dst[i] = -1;
        }
        isize=0;
    }
    bool valid() const { return m_decoded; }

    virtual void print_insn( FILE *fp ) const
    {
        fprintf(fp," [inst @ pc=0x%04x] ", pc );
    }
    virtual void print_insn2( ) const {
        printf(" [inst @ pc=0x%04x] ", pc );
    }
    bool is_load() const { return (op == LOAD_OP || memory_op == memory_load); }
    bool is_store() const { return (op == STORE_OP || memory_op == memory_store); }
    unsigned get_num_operands() const {return num_operands;}
    unsigned get_num_regs() const {return num_regs;}
    void set_num_regs(unsigned num) {num_regs=num;}
    void set_num_operands(unsigned num) {num_operands=num;}
    void set_bar_id(unsigned id) {bar_id=id;}
    void set_bar_count(unsigned count) {bar_count=count;}

    address_type pc;        // program counter address of instruction
    unsigned isize;         // size of instruction in bytes
    op_type op;             // opcode (uarch visible)

    barrier_type bar_type;
    reduction_type red_type;
    unsigned bar_id;
    unsigned bar_count;

    types_of_operands oprnd_type;     // code (uarch visible) identify if the operation is an interger or a floating point
    special_ops sp_op;           // code (uarch visible) identify if int_alu, fp_alu, int_mul ....
    operation_pipeline op_pipe;  // code (uarch visible) identify the pipeline of the operation (SP, SFU or MEM)
    mem_operation mem_op;        // code (uarch visible) identify memory type
    _memory_op_t memory_op; // memory_op used by ptxplus
    unsigned num_operands;
    unsigned num_regs; // count vector operand as one register operand

    address_type reconvergence_pc; // -1 => not a branch, -2 => use function return address

    unsigned out[4];
    unsigned in[4];
    unsigned char is_vectorin;
    unsigned char is_vectorout;
    int pred; // predicate register number
    int ar1, ar2;
    // register number for bank conflict evaluation
    struct {
        int dst[MAX_REG_OPERANDS];
        int src[MAX_REG_OPERANDS];
    } arch_reg;
    //int arch_reg[MAX_REG_OPERANDS]; // register number for bank conflict evaluation
    unsigned latency; // operation latency
    unsigned initiation_interval;

    unsigned data_size; // what is the size of the word being operated on?
    memory_space_t space;
    cache_operator_type cache_op;

protected:
    bool m_decoded;
    virtual void pre_decode() {}
};

enum divergence_support_t {
   POST_DOMINATOR = 1,
   NUM_SIMD_MODEL
};

const unsigned MAX_ACCESSES_PER_INSN_PER_THREAD = 8;

class warp_inst_t: public inst_t {
public:
    // constructors
    warp_inst_t()
    {
        m_uid=0;
        m_empty=true;
        m_config=NULL;
        DRSVR_COALESCE = 0;

        mprb_valid = false;
        mprb_transaction_count = 0;
        mprb_issued_transaction_count = 0;
        mprb_serviced_transactions_count = 0;
        mprb_issue_cycle_time = 0;
        initil_issue = true;

        //drsvrObj = tempObj;
    }
    warp_inst_t( const core_config *config)
    {
        m_uid=0;
        assert(config->warp_size<=MAX_WARP_SIZE);
        m_config=config;
        //drsvrObj = tempObj;
        m_empty=true;
        m_isatomic=false;
        m_per_scalar_thread_valid=false;
        m_mem_accesses_created=false;
        m_cache_hit=false;
        m_is_printf=false;
        DRSVR_COALESCE = 0;
        initil_issue = true;
        mprb_valid = false;
        mprb_transaction_count = 0;
        mprb_issued_transaction_count = 0;
        mprb_serviced_transactions_count = 0;
        mprb_issue_cycle_time = 0;

    }
    virtual ~warp_inst_t(){
        //printf("DRSVR! %u\n", DRSVR_COALESCE);
    }

    // modifiers
    void broadcast_barrier_reduction( const active_mask_t& access_mask);
    void do_atomic(bool forceDo=false);
    void do_atomic( const active_mask_t& access_mask, bool forceDo=false );
    void clear()
    {
        m_empty=true;
    }
    void issue( const active_mask_t &mask, unsigned warp_id, unsigned long long cycle, int dynamic_warp_id, unsigned sm_id, DRSVR *drsvrObj)
    {
        // issue after warp_scheduler
        m_warp_active_mask = mask;
        m_warp_issued_mask = mask;
        m_uid = ++sm_next_uid;
        m_warp_id = warp_id;
        m_dynamic_warp_id = dynamic_warp_id;
        m_sm_id = sm_id;
        issue_cycle = cycle;
        cycles = initiation_interval;
        m_cache_hit=false;
        m_empty=false;
        smObj = drsvrObj;
    }
    const active_mask_t & get_active_mask() const
    {
    	return m_warp_active_mask;
    }
    void completed( unsigned long long cycle ) const;  // stat collection: called when the instruction is completed

    void set_addr( unsigned n, new_addr_type addr )
    {
        if( !m_per_scalar_thread_valid ) {
            m_per_scalar_thread.resize(m_config->warp_size);
            m_per_scalar_thread_valid=true;
        }
        m_per_scalar_thread[n].memreqaddr[0] = addr;
    }
    void set_addr( unsigned n, new_addr_type* addr, unsigned num_addrs )
    {
        if( !m_per_scalar_thread_valid ) {
            m_per_scalar_thread.resize(m_config->warp_size);
            m_per_scalar_thread_valid=true;
        }
        assert(num_addrs <= MAX_ACCESSES_PER_INSN_PER_THREAD);
        for(unsigned i=0; i<num_addrs; i++)
            m_per_scalar_thread[n].memreqaddr[i] = addr[i];
    }

    struct transaction_info {
        std::bitset<4> chunks; // bitmask: 32-byte chunks accessed
        mem_access_byte_mask_t bytes;
        active_mask_t active; // threads in this transaction

        bool test_bytes(unsigned start_bit, unsigned end_bit) {
           for( unsigned i=start_bit; i<=end_bit; i++ )
              if(bytes.test(i))
                 return true;
           return false;
        }
    };

    void generate_mem_accesses();
    void memory_coalescing_arch_13( bool is_write, mem_access_type access_type );
    void memory_coalescing_arch_13_atomic( bool is_write, mem_access_type access_type );
    void memory_coalescing_arch_13_reduce_and_send( bool is_write, mem_access_type access_type, const transaction_info &info, new_addr_type addr, unsigned segment_size );

    void add_callback( unsigned lane_id,
                       void (*function)(const class inst_t*, class ptx_thread_info*),
                       const inst_t *inst,
                       class ptx_thread_info *thread,
                       bool atomic)
    {
        if( !m_per_scalar_thread_valid ) {
            m_per_scalar_thread.resize(m_config->warp_size);
            m_per_scalar_thread_valid=true;
            if(atomic) m_isatomic=true;
        }
        m_per_scalar_thread[lane_id].callback.function = function;
        m_per_scalar_thread[lane_id].callback.instruction = inst;
        m_per_scalar_thread[lane_id].callback.thread = thread;
    }
    void set_active( const active_mask_t &active );

    void clear_active( const active_mask_t &inactive );
    void set_not_active( unsigned lane_id );

    // accessors
    virtual void print_insn(FILE *fp) const
    {
        fprintf(fp," [inst @ pc=0x%04x] ", pc );
        for (int i=(int)m_config->warp_size-1; i>=0; i--)
            fprintf(fp, "%c", ((m_warp_active_mask[i])?'1':'0') );
    }
    bool active( unsigned thread ) const { return m_warp_active_mask.test(thread); }
    unsigned active_count() const { return m_warp_active_mask.count(); }
    unsigned issued_count() const { assert(m_empty == false); return m_warp_issued_mask.count(); }  // for instruction counting
    bool empty() const { return m_empty; }
    unsigned warp_id() const
    {
        assert( !m_empty );
        return m_warp_id;
    }

    unsigned get_warp_id() const
    {
        if (m_empty){
            return -1;
        }
        else {
            return m_warp_id;
        }
    }

    unsigned dynamic_warp_id() const
    {
        assert( !m_empty );
        return m_dynamic_warp_id;
    }
    bool has_callback( unsigned n ) const
    {
        return m_warp_active_mask[n] && m_per_scalar_thread_valid &&
            (m_per_scalar_thread[n].callback.function!=NULL);
    }
    new_addr_type get_addr( unsigned n ) const
    {
        assert( m_per_scalar_thread_valid );
        return m_per_scalar_thread[n].memreqaddr[0];
    }

    bool isatomic() const { return m_isatomic; }

    unsigned warp_size() const { return m_config->warp_size; }

    bool accessq_empty() const { return m_accessq.empty(); }
    unsigned accessq_count() const { return m_accessq.size(); }
    const mem_access_t &accessq_back() { return m_accessq.back(); }
    void accessq_pop_back() { m_accessq.pop_back(); }
    void accessq_clear(){ m_accessq.clear(); }


    void accessq_push_back(mem_access_t input){
        m_accessq.push_back(input);
    }


    std::list<mem_access_t> mprb_get_accessq_list(){
        return m_accessq;
    }

    void mprb_add_accessq_list(std::list<mem_access_t> input){

        m_accessq.clear();
        std::list<mem_access_t>::iterator it;
        for (it = input.begin(); it!= input.end(); it++){
            m_accessq.push_back(*it);
        }

    }



    void accessq_print(){
        unsigned i = 0;
        printf("\n-------------------------- Memory Access Queue  SIZE:%u---------------------------\n",m_accessq.size());
        for (std::list<mem_access_t>::iterator it=m_accessq.begin(); it != m_accessq.end(); ++it){
            printf("m_queue[%u]: ",i);
            (*it).print2();
            printf("\t[%u;%u;%u] ; Active Mask:%s/%s \n"
                    ,m_warp_id
                    ,m_sm_id
                    ,pc
                    ,(*it).get_warp_mask().to_string().c_str()
                    ,m_warp_active_mask.to_string().c_str()
            );
            i++;

        }
        printf("------------------------------------------------------------------------------------\n");
    }

    void accessq_print_toFile(FILE* statFile){
        unsigned i = 0;
        fprintf(statFile,"\n-------------------------- Memory Access Queue  SIZE:%u---------------------------\n",m_accessq.size());
        for (std::list<mem_access_t>::iterator it=m_accessq.begin(); it != m_accessq.end(); ++it){
            fprintf(statFile,"m_queue[%u]: ",i);
            (*it).print2_toFile(statFile);
            fprintf(statFile,"\t[%u;%u;%u] ; Active Mask:%s/%s \n"
                    ,m_warp_id
                    ,m_sm_id
                    ,pc
                    ,(*it).get_warp_mask().to_string().c_str()
                    ,m_warp_active_mask.to_string().c_str()
            );
            i++;

        }
        fprintf(statFile,"------------------------------------------------------------------------------------\n");
    }




    void warp_inst_t_print(bool info, bool out_regs, bool in_regs, bool args, const char *warp_register_name){

        if (info){

            bool row1 = true;
            bool row2 = true;

            printf("------------------------------------------------------------------------------------\n");

            if (row1){
                printf("%s\n[%u;%u;%u] ; op : %s ; oprnd_type : %s ; mem_op : %s ; pipe_op : %s ; issued_cycle_time: %u ; initial_issue: %u ; mprb_valid:%u ;\n"
                        , warp_register_name
                        , m_warp_id, m_sm_id, pc
                        , this->get_uarch_op_t_string(op), this->get_uarch_op_t_string(oprnd_type), this->get_mem_operation_t_string(mem_op), this->get_operation_pipeline_t_string(op_pipe)
                        , mprb_issue_cycle_time, initil_issue, mprb_valid

                );
            }

            if (row2){
                printf("interval: %u ; latency: %u ; num_regs_%u; num_operand:%u ; empty:%u ; accessq_size:%u ; Issued: %u/%u Serviced: %u/%u ; space: %s ; obj: %u ;\n"
                        , initiation_interval, latency, num_regs, num_operands, m_empty, m_accessq.size()
                        , mprb_issued_transaction_count, mprb_transaction_count
                        , mprb_serviced_transactions_count, mprb_transaction_count, this->get_memory_space_t(space.get_type()), this
                );
            }
        }

        if (out_regs) {
            for (unsigned i=0; i<4; i++){
                printf("OUT[%u] = %u\t", i, out[i]);
            }
            printf("\n");
        }

        if (in_regs){
            for (unsigned i=0; i<4; i++){
                printf("IN[%u] = %u\t", i, in[i]);
            }
            printf("\n");
        }

        if (args){
            printf("pred: %u ;\t ar1: %u ; \t ; ar2: %u ;\n", pred, ar1, ar2);
        }

        if (info){
            printf("------------------------------------------------------------------------------------\n");
        }

    }

    void warp_inst_t_print_toFile(bool info, bool out_regs, bool in_regs, bool args, const char *warp_register_name, FILE* statFile){

        if (info){

            bool row1 = true;
            bool row2 = true;

            fprintf(statFile,"------------------------------------------------------------------------------------\n");

            if (row1){
                fprintf(statFile,"%s\n[%u;%u;%u] ; op : %s ; oprnd_type : %s ; mem_op : %s ; pipe_op : %s ; issued_cycle_time: %u ; initial_issue: %u ; mprb_valid:%u ;\n"
                        , warp_register_name
                        , m_warp_id, m_sm_id, pc
                        , this->get_uarch_op_t_string(op), this->get_uarch_op_t_string(oprnd_type), this->get_mem_operation_t_string(mem_op), this->get_operation_pipeline_t_string(op_pipe)
                        , mprb_issue_cycle_time, initil_issue, mprb_valid

                );
            }

            if (row2){
                fprintf(statFile,"interval: %u ; latency: %u ; num_regs_%u; num_operand:%u ; empty:%u ; accessq_size:%u ; Issued: %u/%u Serviced: %u/%u ; space: %s ; obj: %u ;\n"
                        , initiation_interval, latency, num_regs, num_operands, m_empty, m_accessq.size()
                        , mprb_issued_transaction_count, mprb_transaction_count
                        , mprb_serviced_transactions_count, mprb_transaction_count, this->get_memory_space_t(space.get_type()), this
                );
            }
        }

        if (out_regs) {
            for (unsigned i=0; i<4; i++){
                fprintf(statFile,"OUT[%u] = %u\t", i, out[i]);
            }
            fprintf(statFile,"\n");
        }

        if (in_regs){
            for (unsigned i=0; i<4; i++){
                fprintf(statFile,"IN[%u] = %u\t", i, in[i]);
            }
            fprintf(statFile,"\n");
        }

        if (args){
            fprintf(statFile,"pred: %u ;\t ar1: %u ; \t ; ar2: %u ;\n", pred, ar1, ar2);
        }

        if (info){
            fprintf(statFile,"------------------------------------------------------------------------------------\n");
        }

    }







    const char* get_special_operations_t_string (unsigned special_operations_t) {

        const char *operation_Names[] = {
                "OTHER_OP",
                "INT__OP",
                "INT_MUL24_OP",
                "INT_MUL32_OP",
                "INT_MUL_OP",
                "INT_DIV_OP",
                "FP_MUL_OP",
                "FP_DIV_OP",
                "FP__OP",
                "FP_SQRT_OP",
                "FP_LG_OP",
                "FP_SIN_OP",
                "FP_EXP_OP"
        };

        return (operation_Names[special_operations_t]);
    }


    const char* get_operation_pipeline_t_string (unsigned operation_pipeline_t) {
        if (operation_pipeline_t>3){
            printf("operation_pipeline_t: %u ;\n", operation_pipeline_t);
            operation_pipeline_t = 0;
        }
        const char *operation_Names[] = {
                "UNKOWN_OP",
                "SP__OP",
                "SFU__OP",
                "MEM__OP"
        };
        return (operation_Names[operation_pipeline_t]);
    }

    const char* get_mem_operation_t_string (unsigned mem_operation_t) {

        if (mem_operation_t>1){
            printf("mem_operation_t :%u \n",mem_operation_t);
            mem_operation_t = 0;
        }

        const char *operation_Names[] = {
                "NOT_TEX",
                "TEX"
        };

        return (operation_Names[mem_operation_t]);
    }

    const char* get_uarch_op_t_string (unsigned uarch_op_t) {

        if (uarch_op_t == -1){
            return "NO_OP";
        }

        if (uarch_op_t>10){
            printf("uarch_op_t : %u ;\n", uarch_op_t);
            uarch_op_t = 0;
        }

        const char *operation_Names[] = {
                "NO_OP",
                "ALU_OP",
                "SFU_OP",
                "ALU_SFU_OP",
                "LOAD_OP",
                "STORE_OP",
                "BRANCH_OP",
                "BARRIER_OP",
                "MEMORY_BARRIER_OP",
                "CALL_OPS",
                "RET_OPS"
        };

        return (operation_Names[uarch_op_t]);
    }


    const char* get_memory_space_t (unsigned memory_space_t_in) {

        if (memory_space_t_in>12){
            printf("memory_space_t_in : %u ;\n", memory_space_t_in);
            memory_space_t_in = 0;
        }


        const char *space_names[] = {
                "undefined_space",
                "reg_space",
                "local_space",
                "shared_space",
                "param_space_unclassified",
                "param_space_kernel",  /* global to all threads in a kernel : read-only */
                "param_space_local",   /* local to a thread : read-writable */
                "const_space",
                "tex_space",
                "surf_space",
                "global_space",
                "generic_space",
                "instruction_space"
        };

        return (space_names[memory_space_t_in]);
    }

    const char* get_uarch_operand_type_t_string (unsigned uarch_operand_type_t) {

        if (uarch_operand_type_t == -1){
            return "UN_OP";
        }

        const char *uarch_operand_type_t_names[] = {
                "UN_OP",
                "INT_OP",
                "FP_OP"
        };

        return (uarch_operand_type_t_names[uarch_operand_type_t]);
    }


    bool dispatch_delay()
    {
        if( cycles > 0 )
            cycles--;
        return cycles > 0;
    }

    bool has_dispatch_delay(){
    	return cycles > 0;
    }

    void print( FILE *fout ) const;
    unsigned get_uid() const { return m_uid; }


    void mprb_transaction_issued(){
        mprb_issued_transaction_count++;
    }

    void mprb_transaction_serviced(){
        mprb_serviced_transactions_count++;
    }

    unsigned mprb_get_transactionCount(){
        return mprb_transaction_count;
    }

    unsigned mprb_get_issued_transactionCount(){
        return mprb_issued_transaction_count;
    }

    unsigned mprb_get_serviced_transactionCount(){
        return mprb_serviced_transactions_count;
    }

    void mprb_set_transactionCount(unsigned long long input_cycle){
        mprb_transaction_count = accessq_count();
        mprb_serviced_transactions_count = 0;
        mprb_issued_transaction_count = 0;
        mprb_issue_cycle_time = input_cycle;
        mprb_valid = true;
    }





    bool mprb_all_transactions_done(){
        if ( (mprb_valid) && (mprb_transaction_count>0) ){
            if (mprb_serviced_transactions_count == mprb_transaction_count){
                return true;
            }
        }
        return false;
    }


    bool mprb_all_transactions_issued(){
        if ( (mprb_valid) && (mprb_transaction_count>0) ){
            if (mprb_issued_transaction_count == mprb_transaction_count){
                return true;
            }
        }
        return false;
    }

    unsigned long long mprb_get_issue_time(){
        return  mprb_issue_cycle_time;
    }


    void setInitial_issue(){
        initil_issue = true;
    }

    void resetInitial_issue(){
        initil_issue = false;
    }

    bool isInitial(){
        return initil_issue;
    }

    bool mprb_isValid(){
        return mprb_valid;
    }

    void mprb_makeValid(){
        mprb_valid = true;
    }

    void mprb_makeInValid(){
        mprb_valid = false;
    }

protected:

    unsigned m_uid;
    bool m_empty;
    bool m_cache_hit;
    unsigned long long issue_cycle;
    unsigned cycles; // used for implementing initiation interval delay
    bool m_isatomic;
    bool m_is_printf;
    unsigned m_warp_id;
    unsigned m_dynamic_warp_id;
    unsigned m_sm_id;
    const core_config *m_config;
    active_mask_t m_warp_active_mask; // dynamic active mask for timing model (after predication)
    active_mask_t m_warp_issued_mask; // active mask at issue (prior to predication test) -- for instruction counting

    struct per_thread_info {
        per_thread_info() {
            for(unsigned i=0; i<MAX_ACCESSES_PER_INSN_PER_THREAD; i++)
                memreqaddr[i] = 0;
        }
        dram_callback_t callback;
        new_addr_type memreqaddr[MAX_ACCESSES_PER_INSN_PER_THREAD]; // effective address, upto 8 different requests (to support 32B access in 8 chunks of 4B each)
    };
    bool m_per_scalar_thread_valid;
    std::vector<per_thread_info> m_per_scalar_thread;
    bool m_mem_accesses_created;

    std::list<mem_access_t> m_accessq;

    static unsigned sm_next_uid;


    bool initil_issue;
    bool mprb_valid;
    unsigned mprb_transaction_count;
    unsigned mprb_serviced_transactions_count;
    unsigned mprb_issued_transaction_count;
    unsigned long long mprb_issue_cycle_time;




};

void move_warp( warp_inst_t *&dst, warp_inst_t *&src );

size_t get_kernel_code_size( class function_info *entry );

/*
 * This abstract class used as a base for functional and performance and simulation, it has basic functional simulation
 * data structures and procedures.
 */
class core_t {
    public:
        core_t( gpgpu_sim *gpu,
                kernel_info_t *kernel,
                unsigned warp_size,
                unsigned threads_per_shader )
            : m_gpu( gpu ),
              m_kernel( kernel ),
              m_simt_stack( NULL ),
              m_thread( NULL ),
              m_warp_size( warp_size )
        {
            m_warp_count = threads_per_shader/m_warp_size;
            // Handle the case where the number of threads is not a
            // multiple of the warp size
            if ( threads_per_shader % m_warp_size != 0 ) {
                m_warp_count += 1;
            }
            assert( m_warp_count * m_warp_size > 0 );
            m_thread = ( ptx_thread_info** )
                     calloc( m_warp_count * m_warp_size,
                             sizeof( ptx_thread_info* ) );
            initilizeSIMTStack(m_warp_count,m_warp_size);

            for(unsigned i=0; i<MAX_CTA_PER_SHADER; i++){
            	for(unsigned j=0; j<MAX_BARRIERS_PER_CTA; j++){
            		reduction_storage[i][j]=0;
            	}
            }

        }
        virtual ~core_t() { free(m_thread); }
        virtual void warp_exit( unsigned warp_id ) = 0;
        virtual bool warp_waiting_at_barrier( unsigned warp_id ) const = 0;
        virtual void checkExecutionStatusAndUpdate(warp_inst_t &inst, unsigned t, unsigned tid)=0;
        class gpgpu_sim * get_gpu() {return m_gpu;}
        void execute_warp_inst_t(warp_inst_t &inst, unsigned warpId =(unsigned)-1);
        bool  ptx_thread_done( unsigned hw_thread_id ) const ;
        void updateSIMTStack(unsigned warpId, warp_inst_t * inst);
        void initilizeSIMTStack(unsigned warp_count, unsigned warps_size);
        void deleteSIMTStack();
        warp_inst_t getExecuteWarp(unsigned warpId);
        void get_pdom_stack_top_info( unsigned warpId, unsigned *pc, unsigned *rpc ) const;
        kernel_info_t * get_kernel_info(){ return m_kernel;}
        unsigned get_warp_size() const { return m_warp_size; }
        void and_reduction(unsigned ctaid, unsigned barid, bool value) { reduction_storage[ctaid][barid] &= value; }
        void or_reduction(unsigned ctaid, unsigned barid, bool value) { reduction_storage[ctaid][barid] |= value; }
        void popc_reduction(unsigned ctaid, unsigned barid, bool value) { reduction_storage[ctaid][barid] += value;}
        unsigned get_reduction_value(unsigned ctaid, unsigned barid) {return reduction_storage[ctaid][barid];}
    protected:
        class gpgpu_sim *m_gpu;
        kernel_info_t *m_kernel;
        simt_stack  **m_simt_stack; // pdom based reconvergence context for each warp
        class ptx_thread_info ** m_thread;
        unsigned m_warp_size;
        unsigned m_warp_count;
        unsigned reduction_storage[MAX_CTA_PER_SHADER][MAX_BARRIERS_PER_CTA];
};


//register that can hold multiple instructions.
class register_set {
public:
	register_set(unsigned num, const char* name){
		for( unsigned i = 0; i < num; i++ ) {
			regs.push_back(new warp_inst_t());
		}
		m_name = name;
	}
	bool has_free(){
		for( unsigned i = 0; i < regs.size(); i++ ) {
			if( regs[i]->empty() ) {
				return true;
			}
		}
		return false;
	}
	bool has_ready(){
		for( unsigned i = 0; i < regs.size(); i++ ) {
			if( not regs[i]->empty() ) {
				return true;
			}
		}
		return false;
	}

	void move_in( warp_inst_t *&src ){
		warp_inst_t** free = get_free();
		move_warp(*free, src);
	}
	//void copy_in( warp_inst_t* src ){
		//   src->copy_contents_to(*get_free());
		//}
	void move_out_to( warp_inst_t *&dest ){
		warp_inst_t **ready=get_ready();
		move_warp(dest, *ready);
	}

	warp_inst_t** get_ready(){
        //printf("GET READY! regs.size():%u ; name:%s ;\n ",regs.size(), m_name);
		warp_inst_t** ready;
		ready = NULL;
		for( unsigned i = 0; i < regs.size(); i++ ) {
			if( not regs[i]->empty() ) {
				if( ready and (*ready)->get_uid() < regs[i]->get_uid() ) {
					// ready is oldest
				} else {
					ready = &regs[i];
				}
			}
		}
		return ready;
	}

	void print(FILE* fp) const{
		fprintf(fp, "%s : @%p\n", m_name, this);
		for( unsigned i = 0; i < regs.size(); i++ ) {
			fprintf(fp, "     ");
			regs[i]->print(fp);
			fprintf(fp, "\n");
		}
	}

    void print2(bool detail) const{
        for( unsigned i = 0; i < regs.size(); i++ ) {
            if (detail){
                if (regs[i]->get_warp_id()!=-1){
                    printf("--------------------------------------------------------------------------------\n");
                    printf("reg[%u]:\n", i);
                    regs[i]->warp_inst_t_print(true, true, true, true, m_name);
                    if ( regs[i]->accessq_count()>0){
                        regs[i]->accessq_print();
                    }
                    printf("--------------------------------------------------------------------------------\n");
                }
            }
            else {
                if (regs[i]->get_warp_id()!=-1){
                    printf("--------------------------------------------------------------------------------\n");
                    printf("reg[%u] = %u \n", i, regs[i]->get_warp_id());
                    printf("--------------------------------------------------------------------------------\n");
                }
            }
        }
    }


    void print2_toFile(bool detail, FILE* statFile) const{
        for( unsigned i = 0; i < regs.size(); i++ ) {
            if (detail){
                if (regs[i]->get_warp_id()!=-1){
                    fprintf(statFile,"--------------------------------------------------------------------------------\n");
                    fprintf(statFile,"reg[%u]:\n", i);
                    regs[i]->warp_inst_t_print_toFile(true, true, true, true, m_name, statFile);
                    if ( regs[i]->accessq_count()>0){
                        regs[i]->accessq_print_toFile(statFile);
                    }
                    fprintf(statFile,"--------------------------------------------------------------------------------\n");
                }
            }
            else {
                if (regs[i]->get_warp_id()!=-1){
                    fprintf(statFile,"--------------------------------------------------------------------------------\n");
                    fprintf(statFile,"reg[%u] = %u \n", i, regs[i]->get_warp_id());
                    fprintf(statFile,"--------------------------------------------------------------------------------\n");
                }
            }
        }
    }


	warp_inst_t ** get_free(){
		for( unsigned i = 0; i < regs.size(); i++ ) {
			if( regs[i]->empty() ) {
				return &regs[i];
			}
		}
		assert(0 && "No free registers found");
		return NULL;
	}


    unsigned getSize(){
        return regs.size();
    }

    std::string getName(){
        std::string unitName = m_name;
        return unitName;
    }

    void printInfo(){
        printf("pipeRegs Size: %u ;  Name: %s\n", regs.size(), m_name);
    }

private:
	std::vector<warp_inst_t*> regs;
	const char* m_name;
};




class MPRB_TRANSACTION_DATA{

public:

    MPRB_TRANSACTION_DATA(unsigned warp_id_in, unsigned long long issue_time_in, mem_access_t transaction_in ){
        warp_id = warp_id_in;
        issue_time = issue_time_in;
        transaction = transaction_in;
    }

    unsigned get_warp_id(){
        return warp_id;
    }

    unsigned long long get_issue_time(){
        return issue_time;
    }

    mem_access_t get_transaction(){
        return transaction;
    }

private:

    unsigned warp_id;
    unsigned long long issue_time;
    mem_access_t transaction;

};



class MPRB {


public:

    MPRB() {

        // Input Buffer
        for (unsigned i = 0; i < inputBufferSize; i++) {
            inputBuffer.push_back(new warp_inst_t());
        }

        // Warp Queue
        for (unsigned i = 0; i < warpQueueSize; i++) {
            warpsQueue.push_back(new warp_inst_t());
        }
        warpsObjQueue.clear();

        // Transaction Queue
        transactionsQueue.clear();
        transactionObjQueue.clear();


    }


    // Input Buffer Functions
    bool buffer_has_free() {
        for (unsigned i = 0; i < inputBuffer.size(); i++) {
            if (inputBuffer[i]->empty()) {
                return true;
            }
        }
        return false;
    }

    bool buffer_has_ready() {
        for (unsigned i = 0; i < inputBuffer.size(); i++) {
            if (not inputBuffer[i]->empty()) {
                return true;
            }
        }
        return false;
    }


    warp_inst_t **buffer_get_free() {
        for (unsigned i = 0; i < inputBuffer.size(); i++) {
            if (inputBuffer[i]->empty()) {
                return &inputBuffer[i];
            }
        }
        assert(0 && "No free bufferQueue MPRB found!");
        return NULL;
    }

    warp_inst_t **buffer_get_ready() {
        for (unsigned i = 0; i < inputBuffer.size(); i++) {
            if (not inputBuffer[i]->empty()) {
                return &inputBuffer[i];
            }
        }
        assert(0 && "No ready bufferQueue MPRB found!");
        return NULL;
    }


    void fill_in_inputBuffer(warp_inst_t *&src) {
        warp_inst_t **free = buffer_get_free();
        move_warp(*free, src);
    }


    void get_out_inputBuffer(warp_inst_t *&dest) {
        //warp_inst_t **ready=buffer_get_ready();
        warp_inst_t **ready = &inputBuffer[0];
        move_warp(dest, *ready);
        for (unsigned i = 0; (i + 1) < inputBuffer.size(); i++) {
            if (not inputBuffer[i + 1]->empty()) {
                move_warp(inputBuffer[i], inputBuffer[i + 1]);
                inputBuffer[i + 1] = new warp_inst_t();
            } else {
                inputBuffer[i] = new warp_inst_t();
            }
        }
    }


    void print_inputBuffer() {
        for (unsigned i = 0; i < inputBuffer.size(); i++) {
            std::ostringstream oss;
            oss << "input_buffer[" << i << "]";
            std::string name = oss.str();
            if (inputBuffer[i]->empty()) {
                printf("input_buffer[%u] = empty ; \n", i);
            } else {
                inputBuffer[i]->warp_inst_t_print(true, false, false, false, name.c_str());
            }
        }
    }




    // WarpQueue Buffer Functions

    bool WarpQueue_has_free() {
        for (unsigned i = 0; i < warpsQueue.size(); i++) {
            if (warpsQueue[i]->empty()) {
                return true;
            }
        }
        return false;
    }

    bool WarpQueue_has_ready() {
        for (unsigned i = 0; i < warpsQueue.size(); i++) {
            if (not warpsQueue[i]->empty()) {
                return true;
            }
        }
        return false;
    }


    warp_inst_t **WarpQueue_get_free() {
        for (unsigned i = 0; i < warpsQueue.size(); i++) {
            if (warpsQueue[i]->empty()) {
                return &warpsQueue[i];
            }
        }
        assert(0 && "No free bufferQueue MPRB found!");
        return NULL;
    }

    warp_inst_t **WarpQueue_ready() {
        for (unsigned i = 0; i < warpsQueue.size(); i++) {
            if (not warpsQueue[i]->empty()) {
                return &warpsQueue[i];
            }
        }
        assert(0 && "No ready bufferQueue MPRB found!");
        return NULL;
    }


    void fill_in_warpsQueue(warp_inst_t *&src) {
        warp_inst_t **free = WarpQueue_get_free();
        move_warp(*free, src);
    }

    void get_out_warpsQueue(warp_inst_t *&dest) {
        //warp_inst_t **ready=buffer_get_ready();
        warp_inst_t **ready = &warpsQueue[0];
        move_warp(dest, *ready);
        for (unsigned i = 0; (i + 1) < warpsQueue.size(); i++) {
            if (not warpsQueue[i + 1]->empty()) {
                move_warp(warpsQueue[i], warpsQueue[i + 1]);
                warpsQueue[i + 1] = new warp_inst_t();
            } else {
                warpsQueue[i] = new warp_inst_t();
            }
        }
    }


    void print_warpsQueue() {
        printf("MPRB WarpsQueue! Size: %u\n", warpsQueue.size());
        printf("------------------------------------------------------------\n");
        for (unsigned i = 0; i < warpsQueue.size(); i++) {
            std::ostringstream oss;
            oss << "warpsQueue[" << i << "]";
            std::string name = oss.str();
            if (warpsQueue[i]->empty()) {
                printf("warpsQueue[%u] = empty ; \n", i);
            } else {
                warpsQueue[i]->warp_inst_t_print(true, false, false, false, name.c_str());
            }
        }
        printf("------------------------------------------------------------\n");
    }


    void filter_split_transactions_warp(warp_inst_t *&src) {

        assert(!src->empty());

        unsigned warp_id = src->get_warp_id();

        std::list<mem_access_t> m_accessq = src->mprb_get_accessq_list();

        std::list<mem_access_t>::iterator it;

        for (it = m_accessq.begin(); it != m_accessq.end(); it++) {
            (*it).mprb_set_warp_id(warp_id);
            transactionsQueue.push_back(*it);
        }

        if (WarpQueue_has_free()) {
            warp_inst_t **free = WarpQueue_get_free();
            **free = *src;
            (*free)->mprb_set_transactionCount(0);
            (*free)->accessq_clear();
        }

        printf("\n\n");
        print_transaction_queue();
        printf("\n\n");
        print_warpsQueue();

        printf("\n\n");
        merger(src);
        printf("\n\n");

        print_transaction_queue();
        printf("\n\n");
        print_warpsQueue();
        printf("\n\n");


    }


    warp_inst_t* search_warpsQueue(unsigned warp_id){

        warp_inst_t * foundWarp = new warp_inst_t();

        for (unsigned i=0; i<warpsQueue.size(); i++){
            if (not warpsQueue[i]->empty()){
                if (warpsQueue[i]->get_warp_id() == warp_id){
                    warpsQueue[i]->warp_inst_t_print(true, false, false, false, "search");
                    foundWarp = warpsQueue[i];
                    return foundWarp;
                }
            }
        }

        return foundWarp;
    }

    void remove_doneWarp(unsigned warp_id){
        for (unsigned i=0; i<warpsQueue.size(); i++){
            if (warpsQueue[i]->get_warp_id() == warp_id){
                for (unsigned j=i; j+1<warpsQueue.size(); j++){
                    warpsQueue[j] = warpsQueue[j+1];
                }
                warpsQueue.pop_back();
                return;
            }
        }
    }



    mem_access_t transaction_pop_front(){

        assert(transactionsQueue.size()>0);

        mem_access_t gready = transactionsQueue.at(0);

        for (unsigned i=0; (i+1)<transactionsQueue.size(); i++){
            transactionsQueue[i] = transactionsQueue[i+1];
        }

        transactionsQueue.pop_back();

        return gready;

    }

    warp_inst_t* search_ready_warp(unsigned long long issue_time, unsigned warp_id){

        for (unsigned i=0; i<warpsObjQueue.size(); i++){
            if ( (warpsObjQueue.at(i)->get_warp_id() == warp_id) && (warpsObjQueue.at(i)->mprb_get_issue_time()==issue_time) ){
                return warpsObjQueue.at(i);
            }
        }


        printf("warp_id:%u ; issue_time:%llu ; transactionQueueSize:%u ; warpObjQueue.size:%u ;\n", warp_id, issue_time, transactionObjQueue.size(), warpsObjQueue.size());

        print_warpObj_queue();

        //print_transactionObj_queue();
        //print_warpObj_queue();

        printf("Something is Wrong, we could not find the requested warp from the warpsQueue!\n");
        assert(false);
    }

    void remove_all_issued_warps(){
        for (unsigned i=0; i<warpsObjQueue.size(); i++){
            if (warpsObjQueue.at(i)->mprb_all_transactions_issued()){
                for (unsigned j=i; (j+1)<warpsObjQueue.size(); j++){
                    warpsObjQueue[j] = warpsObjQueue[j+1];
                }
                warpsObjQueue.pop_back();
            }
        }
    }



    MPRB_TRANSACTION_DATA* transactionObj_pop_front(){

        assert(transactionObjQueue.size()>0);

        MPRB_TRANSACTION_DATA* gready_transaction = transactionObjQueue.at(0);

        for (unsigned i=0; (i+1)<transactionObjQueue.size(); i++){
            transactionObjQueue[i] = transactionObjQueue[i+1];
        }

        transactionObjQueue.pop_back();

        return gready_transaction;

    }


    void set_output_ready_warp(){

        assert(warpsObjQueue.size()>0);
        assert(transactionObjQueue.size()>0);

        MPRB_TRANSACTION_DATA* transaction_obj = transactionObj_pop_front();


        unsigned long long issue_time = transaction_obj->get_issue_time();
        unsigned warp_id = transaction_obj->get_warp_id();

        transaction_obj->get_transaction().print2();



        warp_inst_t* readyWarp = search_ready_warp(issue_time, warp_id);

        readyWarp->mprb_transaction_issued();

        mem_access_t readyTransaction = transaction_obj->get_transaction();

        output_buffer = (*readyWarp);

        output_buffer.accessq_push_back(readyTransaction);
    }



    warp_inst_t pop_ready_warp(){
        assert(can_pop_warp());
        set_output_ready_warp();
        remove_all_issued_warps();
        if (output_buffer.mprb_get_issued_transactionCount()==1){
            output_buffer.setInitial_issue();
        }
        else {
            output_buffer.resetInitial_issue();
        }
        return output_buffer;
    }





    void push_ready_warp(warp_inst_t* input_warp){
        assert(warpsObjQueue.size()<warpQueueSize);
        assert(transactionObjQueue.size()<transactionQueueSize);
        assert(input_warp->mprb_isValid());
        input_buffer = new warp_inst_t();
        *input_buffer = *input_warp;
        mprb_split_transactions_warp();
    }

    void mprb_split_transactions_warp() {

        assert(not input_buffer->empty());
        assert(input_buffer->mprb_isValid());

        assert(can_push_warp());

        unsigned warp_id = input_buffer->get_warp_id();
        unsigned long long issue_time = input_buffer->mprb_get_issue_time();
        std::list<mem_access_t> m_accessq = input_buffer->mprb_get_accessq_list();


        while (m_accessq.size()>0){
            mem_access_t transaction_element = m_accessq.front();
            m_accessq.pop_front();
            transaction_element.mprb_set_warp_id(warp_id);
            MPRB_TRANSACTION_DATA* transaction_obj = new MPRB_TRANSACTION_DATA(warp_id, issue_time, transaction_element);
            transactionObjQueue.push_back(transaction_obj);
        }

        input_buffer->accessq_clear();

        warpsObjQueue.push_back(input_buffer);

    }






    bool can_push_warp(){

        if (warpsObjQueue.size()<warpQueueSize){
            if (transactionObjQueue.size()<transactionQueueSize){
                return true;
            }
        }

        return false;
    }

    bool can_pop_warp(){

        if (warpsObjQueue.size()>0){
            if (transactionObjQueue.size()>0){
                return true;
            }
        }

        return false;
    }







    void merger(warp_inst_t *&src) {

        mem_access_t readyTransaction = transaction_pop_front();

        unsigned warp_id = readyTransaction.mprb_get_warp_id();

        warp_inst_t *drsvrWarp = search_warpsQueue(warp_id);
        assert(drsvrWarp);

        drsvrWarp->accessq_push_back(readyTransaction);

        drsvrWarp->mprb_transaction_serviced();

        if (drsvrWarp->mprb_all_transactions_done()){
            remove_doneWarp(warp_id);
        }

        drsvrWarp->warp_inst_t_print(true, false, false, false, "drsvrWarp_merger");

        src = drsvrWarp;
    }


    void print_transactionObj_queue(){
        printf("------------------------------------------------------------\n");
        printf("MPRB Transaction Queue! Size: %u\n", transactionObjQueue.size());
        printf("------------------------------------------------------------\n");
        for (unsigned i=0; i<transactionObjQueue.size(); i++){
            printf("transaction_queue[%u] : ", i);
            transactionObjQueue.at(i)->get_transaction().print2();
            printf(" issued_time: %llu ;", transactionObjQueue.at(i)->get_issue_time());
            printf("\n");
        }
        printf("------------------------------------------------------------\n");
    }


    void print_transactionObj_queue_toFile(FILE* statFile){
        fprintf(statFile,"------------------------------------------------------------\n");
        fprintf(statFile,"MPRB Transaction Queue! Size: %u\n", transactionObjQueue.size());
        fprintf(statFile,"------------------------------------------------------------\n");
        for (unsigned i=0; i<transactionObjQueue.size(); i++){
            fprintf(statFile,"transaction_queue[%u] : ", i);
            transactionObjQueue.at(i)->get_transaction().print2_toFile(statFile);
            fprintf(statFile," issued_time: %llu ;", transactionObjQueue.at(i)->get_issue_time());
            fprintf(statFile,"\n");
        }
        fprintf(statFile,"------------------------------------------------------------\n");
    }


    void print_warpObj_queue(){
        printf("------------------------------------------------------------\n");
        printf("MPRB WarpsQueue! Size: %u\n", warpsObjQueue.size());
        printf("------------------------------------------------------------\n");
        for (unsigned i=0; i<warpsObjQueue.size(); i++){
            std::ostringstream oss;
            oss << "warpsQueue[" << i << "]";
            std::string name = oss.str();
            if (warpsObjQueue[i]->mprb_isValid()){
                warpsObjQueue[i]->warp_inst_t_print(true, false, false, false, name.c_str());
            }
            else {
                printf("Something is Wrong!");
                assert(false);
            }

        }
        printf("------------------------------------------------------------\n");
    }


    void print_warpObj_queue_toFile(FILE* statFile){
        fprintf(statFile,"------------------------------------------------------------\n");
        fprintf(statFile,"MPRB WarpsQueue! Size: %u\n", warpsObjQueue.size());
        fprintf(statFile,"------------------------------------------------------------\n");
        for (unsigned i=0; i<warpsObjQueue.size(); i++){
            std::ostringstream oss;
            oss << "warpsQueue[" << i << "]";
            std::string name = oss.str();
            if (warpsObjQueue[i]->mprb_isValid()){
                warpsObjQueue[i]->warp_inst_t_print_toFile(true, false, false, false, name.c_str(), statFile);
            }
            else {
                fprintf(statFile,"Something is Wrong!");
                assert(false);
            }

        }
        fprintf(statFile,"------------------------------------------------------------\n");
    }




    void print_transaction_queue(){
        printf("MPRB Transaction Queue! Size: %u\n", transactionsQueue.size());
        printf("------------------------------------------------------------\n");
        for (unsigned i=0; i<transactionsQueue.size(); i++){
            printf("transaction_queue[%u] :  ", i);
            transactionsQueue.at(i).print2();
            printf("\n");
        }
        printf("------------------------------------------------------------\n");
    }





    enum drainPolicy{
        FIFO,
        LQF,
    };

    enum signature{
        OAWS,
        WARP_ID,
    };

private:

    std::vector<warp_inst_t*> inputBuffer;
    unsigned inputBufferSize = 2;

    std::vector<warp_inst_t*> warpsQueue;
    unsigned warpQueueSize = 10;

    std::vector<mem_access_t> transactionsQueue;
    unsigned transactionQueueSize = 100;


    std::vector<MPRB_TRANSACTION_DATA*> transactionObjQueue;
    std::vector<warp_inst_t*> warpsObjQueue;


    warp_inst_t output_buffer;
    warp_inst_t *input_buffer;

};

class DRSVR{

private:
    unsigned d_sm_id;

    const unsigned warpPerSM = 48;

    std::vector<DRSVRSTATS*> stats_warps_obj_vector;

    std::vector<std::vector<DRSVRSTATS*> > stats_kernels_obj_vector;

    std::vector<DRSVRSTATS*> stats_kernels_obj_global;


    std::vector<DLC*> dlc_obj_vector;

    DRSVRSTATS *global_stats_obj;
    DRSVRSTATS *stats_kernels_obj_global_aggr;

    DLC *global_dlc_obj;
    DLC *global_dlc_obj_aggr;

    OCW_LOGIC *global_ocw_obj;

    unsigned missOnFlight;
    unsigned availableMSHR;
    bool  mshrStatsInitialized = false;

    unsigned OCW_VALUE;
    //bool OCW_VALID;

    unsigned missPrediction;


    FILE *statFile;




public:


    FCLUnit *global_FCL_obj;

    MPRB *global_MPRB_obj;

    DRSVR(unsigned input_sm_id) {

        d_sm_id = input_sm_id;

        //global_stats_obj = new DRSVRSTATS(d_sm_id, warpPerSM);

        global_dlc_obj = new DLC(32,5,128);             // DRSVR WARNING: You should add parameters instead of constants
        // Numbers does not matter anymore since we do not calculate sets manually anymore
        global_dlc_obj_aggr = new DLC(32,5,128);

        global_ocw_obj = new OCW_LOGIC();

        global_FCL_obj = new FCLUnit();

        global_MPRB_obj = new MPRB();

        OCW_VALUE = 2;
        //OCW_VALID = false;

        stats_kernels_obj_global_aggr = new DRSVRSTATS(d_sm_id, warpPerSM);

        this->initialize_stats_warps_obj_vector(false);


        std::ostringstream oss;

        oss << "/home/svali003/testbench/Benchmarks/OAWS_DEBUG/"<<"statFile_"<<input_sm_id<<".drsvr";
        std::string name = oss.str();

        statFile = fopen(name.c_str(),"w");

    }


    void enableFCLDebug(){
        global_FCL_obj->enableDebugMode();
    }


    FILE* getStatFile(){
        return statFile;
    }

    void initialize_stats_warps_obj_vector(bool bankIt){

        if (bankIt){
            stats_kernels_obj_vector.push_back(stats_warps_obj_vector);
            stats_kernels_obj_global.push_back(global_stats_obj);
        }

        stats_warps_obj_vector.clear();

        for (unsigned d_warp_id=0; d_warp_id<warpPerSM; d_warp_id++) {

            DRSVRSTATS *statsObj_temp = new DRSVRSTATS(d_sm_id, d_warp_id);

            stats_warps_obj_vector.push_back(statsObj_temp);

        }

        //delete global_stats_obj;
        global_stats_obj = new DRSVRSTATS(d_sm_id, warpPerSM);


    }

    void printout_dlc_tofile_singleKernel(FILE* dlcFile, unsigned target_kernel_uid){
        if (dlc_obj_vector.size()>0){
            dlc_obj_vector.at(target_kernel_uid)->printTofile_DLC(dlcFile,this->get_sm_id());
        }
    }

    void printout_dlc_tofile_lastKernel(FILE* dlcFile){
        if (dlc_obj_vector.size()>0){
            unsigned lastKernel = dlc_obj_vector.size()-1;
            dlc_obj_vector.at(lastKernel)->printTofile_DLC(dlcFile,this->get_sm_id());
        }
    }


    void printout_dlc_tofile_multiKernel(FILE* dlcFile){

        for (unsigned i=0; i<dlc_obj_vector.size() ; i++){
            this->printout_dlc_tofile_singleKernel(dlcFile,i);
        }

    }

    void kernel_done_bankData(unsigned kernel_id){

        dlc_obj_vector.push_back(global_dlc_obj);
        global_dlc_obj->aggragateDLC(global_dlc_obj_aggr);
        global_dlc_obj = new DLC(32,5,128);

        global_FCL_obj = new FCLUnit();
        global_ocw_obj = new OCW_LOGIC();

        set_OCW_value(2);

        //this->print_histogram_global("FCL");
        this->initialize_stats_warps_obj_vector(true);
    }


    void set_OCW_value(unsigned OCW_IN){
        OCW_VALUE = OCW_IN;
    }

    unsigned get_OCW_value(){
        return (OCW_VALUE);
    }

    unsigned get_sm_id (){
        return d_sm_id;
    }

    void update_PC_histogram(unsigned input_warp_id, unsigned input_pc, std::string histogramName, unsigned input_data){
        stats_warps_obj_vector.at(input_warp_id)->update_PC_Histogram(histogramName, input_pc, input_data);
        global_stats_obj->update_PC_Histogram(histogramName, input_pc, input_data);
        stats_kernels_obj_global_aggr->update_PC_Histogram(histogramName, input_pc, input_data);
    }


    void update_PC_histogram_double(unsigned input_warp_id, unsigned input_pc, std::string histogramName, double input_data){
        stats_warps_obj_vector.at(input_warp_id)->update_PC_Histogram_double(histogramName, input_pc, input_data);
        global_stats_obj->update_PC_Histogram_double(histogramName, input_pc, input_data);
        stats_kernels_obj_global_aggr->update_PC_Histogram_double(histogramName, input_pc, input_data);
    }

    void update_PC_histogram_string(unsigned input_warp_id, unsigned input_pc, std::string histogramName, std::string input_data){
        stats_warps_obj_vector.at(input_warp_id)->update_PC_Histogram_string(histogramName, input_pc, input_data);
        global_stats_obj->update_PC_Histogram_string(histogramName, input_pc, input_data);
        stats_kernels_obj_global_aggr->update_PC_Histogram_string(histogramName, input_pc, input_data);
    }


    void update_histogram(unsigned input_warp_id, std::string histogramName, unsigned input_data){
        stats_warps_obj_vector.at(input_warp_id)->update_histogram(histogramName, input_data);
        global_stats_obj->update_histogram(histogramName, input_data);
        stats_kernels_obj_global_aggr->update_histogram(histogramName, input_data);
    }

    void update_histogram_double(unsigned input_warp_id, std::string histogramName, double input_data){
        stats_warps_obj_vector.at(input_warp_id)->update_histogram_double(histogramName, input_data);
        global_stats_obj->update_histogram_double(histogramName, input_data);
        stats_kernels_obj_global_aggr->update_histogram_double(histogramName,input_data);
    }

    void update_histogram_string(unsigned input_warp_id, std::string histogramName, std::string input_data){
        stats_warps_obj_vector.at(input_warp_id)->update_histogram_string(histogramName, input_data);
        global_stats_obj->update_histogram_string(histogramName, input_data);
        stats_kernels_obj_global_aggr->update_histogram_string(histogramName,input_data);
    }



    void print_histogram(unsigned input_warp_id, std::string histogramName){
        stats_warps_obj_vector.at(input_warp_id)->print_histogram(histogramName);
    }

    void print_DC_histogram(unsigned input_warp_id, std::string histogramName){
        stats_warps_obj_vector.at(input_warp_id)->print_PC_Histogram(histogramName);
    }

    void print_histogram_global(std::string histogramName){
        global_stats_obj->print_histogram(histogramName);
    }

    void print_histogram_global_aggr(std::string histogramName){
        stats_kernels_obj_global_aggr->print_histogram(histogramName);
    }

    void print_pc_histogram_global_aggr(std::string histogramName){
        stats_kernels_obj_global_aggr->print_PC_Histogram(histogramName);
    }


    void print_histogram_global_vector(std::string histogramName){
        for (unsigned i=0; i<stats_kernels_obj_global.size(); i++){
            stats_kernels_obj_global.at(i)->print_histogram(histogramName);
        }
    }

    void print_histogram_vector(std::string histogramName){
        global_stats_obj->print_histogram(histogramName);
        for (unsigned i=0; i<stats_warps_obj_vector.size(); i++){
            stats_warps_obj_vector.at(i)->print_histogram(histogramName);
        }
    }

/*    void update_dlc_table_V2 (unsigned input_warpid, unsigned input_sm_id, unsigned input_pc, std::vector<unsigned> transaction_vector, bool isWrite ){
        if ( (transaction_vector.size()>1) && (!isWrite)) { // Divergent Load

            printf("Divergent Load: Warp ID/ SM ID/PC:[%u;%u;%u] Count:%u\n",input_warpid, input_sm_id, input_pc, transaction_vector.size());
            global_dlc_obj->update_DLC(input_pc, transaction_vector);
        }
    }*/

    void update_dlc_table (unsigned input_warpid, unsigned input_sm_id, unsigned input_pc, unsigned input_inst, unsigned input_acc, unsigned input_set, bool isLoad, unsigned missCount){

        assert(isLoad && (input_acc>1));

        //printf("Divergent Load: Warp ID/ SM ID/PC:[%u;%u;%u]\t",input_warpid, input_sm_id, input_pc);
        //printf("PC:%u ; INST:%u ; ACC:%u ; SET:%u;\n", input_pc ,input_inst ,input_acc ,input_set);
        global_dlc_obj->update_DLC_entry(input_pc,input_inst ,input_acc ,input_set, missCount);

    }

    void print_dlc_table(){
        global_dlc_obj->print_DLC(this->get_sm_id());
        print_mshr_info();
        printf("-------------------------------------------------------------------------------------------------------------\n");
    }

    void print_dlc_table_aggregated(){
        global_dlc_obj_aggr->print_DLC(this->get_sm_id());
        print_mshr_info();
        printf("-------------------------------------------------------------------------------------------------------------\n");
    }


    void print_dlc_table_tofile(FILE *dlcFile){
        global_dlc_obj->printTofile_DLC(dlcFile, (this)->get_sm_id());
        //print_mshr_info();
        fprintf(dlcFile,"-------------------------------------------------------------------------------------------------------------\n");
    }


    void print_dlc_transaction_history(){
        global_dlc_obj->print_transactions_history();
    }

    void update_availableMSHR (unsigned newAvailable){
        availableMSHR = newAvailable;
    }

    unsigned get_availableMSHR (){
        return (availableMSHR);
    }

    void update_missOnFlight (unsigned newMissOnFlight){
        //printf("newMissOnFlight: %u; missOnFlight:%u\n",newMissOnFlight,missOnFlight);
        missOnFlight = newMissOnFlight;
    }

    unsigned get_missOnFlight(){
        return missOnFlight;
    }

    void initialize_mshr_OAWS(unsigned available_in, unsigned missOnFlight_in){
        assert(mshrStatsInitialized==false);
        availableMSHR = available_in;
        missOnFlight = missOnFlight_in;

        printf("DRSVR MSHR OAWS STATUS HAS BEEN Initialized:> availableMSHR = %u ; missOnFlight = %u;\n"
                , availableMSHR, missOnFlight);

    }

    void print_mshr_info(){
        printf("DRSVR MSHR OAWS INFO: availableMSHR = %u ; missOnFlight = %u;\n"
                , availableMSHR, missOnFlight);
    }

    unsigned getMissPredictionCount(){
        return missPrediction;
    }

    bool missPred(unsigned input_PC, unsigned active_threads, unsigned gto_prio){

        bool DRSVRdebug = false;

        unsigned remainingMSHR = availableMSHR - missOnFlight;

        unsigned SMR_Fraction = 2;
        unsigned fixedActive_threads = 1;
        unsigned requiredMSHR = 0;

        fixedActive_threads = active_threads;

        if (active_threads>=SMR_Fraction){
            while (fixedActive_threads%SMR_Fraction!=0){
                fixedActive_threads++;
            }
            //requiredMSHR = (fixedActive_threads/SMR_Fraction);
            requiredMSHR = (fixedActive_threads/SMR_Fraction) + gto_prio;
        }
        else {
            //requiredMSHR = 1;
            requiredMSHR = 1 + gto_prio;
        }


        // ADD CAP

        unsigned capMiss = global_dlc_obj->get_MissCount(input_PC);

        if (requiredMSHR > capMiss){
            requiredMSHR = capMiss;
        }

        missPrediction = requiredMSHR;

        if ( (remainingMSHR<requiredMSHR) ){
            if ( DRSVRdebug ){
                printf("DRSVR MSHR NOT APPROVED! remainingMSHR:%u ; availableMSHR:%u; missOnFlight:%u; activeThreads: %u; requiredMSHR: %u;\n"
                        ,remainingMSHR
                        ,availableMSHR
                        ,missOnFlight
                        ,active_threads
                        ,requiredMSHR);
            }
            return false;
        }
        else {
            if ( DRSVRdebug ){
                printf("DRSVR MSHR APPROVED! remainingMSHR:%u ; availableMSHR:%u; missOnFlight:%u; activeThreads: %u; requiredMSHR: %u;\n"
                        ,remainingMSHR
                        ,availableMSHR
                        ,missOnFlight
                        ,active_threads
                        ,requiredMSHR);
            }
            return true;
        }

    }

    bool oawsApproved(unsigned input_PC, unsigned active_threads, unsigned gto_prio, bool OCW_Approved){
        if (global_dlc_obj->isDivergent(input_PC)){

            // Locality warps should be issued anyway
            if (OCW_Approved){
                missPrediction = 0;
                return true;
            }

/*            unsigned Inst = global_dlc_obj->get_InstOccurance(input_PC);
            unsigned Acc = global_dlc_obj->get_TransactionCounts(input_PC);
            unsigned Set = global_dlc_obj->get_SetTouched(input_PC);
            printf("%u is a dirvergent Load! ActiveThreads: %u/32 ; Inst: %u ; Acc : %u ; Set : %u ;\n"
                    ,input_PC
                    ,active_threads
                    ,Inst
                    ,Acc
                    ,Set
            );*/

            bool missPred_Approved = missPred(input_PC, active_threads, gto_prio);

            return (missPred_Approved);

        }
        else {
            //printf("%u is not a divergent Load!\n", input_PC);
            return true;
        }
    }

    void updateOCW(bool status, unsigned set_in, unsigned acc_in){
        if (global_FCL_obj->isDivergent() && global_FCL_obj->isDone()){
            //printf("DRSVR UPDATE FCL: %u->%u\n", global_ocw_obj->getFCL(), status);
            global_ocw_obj->isLoadFullyCached(status);
            global_ocw_obj->isLoadDivergent(global_FCL_obj->isDivergent());
            global_ocw_obj->updateOCW_SET_ACC(set_in, acc_in);
        }
    }

    unsigned getOCW(bool debugMode){
        return (global_ocw_obj->getOCW(debugMode));
    }


};



#endif // #ifdef __cplusplus

#endif // #ifndef ABSTRACT_HARDWARE_MODEL_INCLUDED
