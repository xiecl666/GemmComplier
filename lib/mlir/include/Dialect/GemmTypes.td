// TODO: 在此定义 Gemm Dialect 的自定义类型
#ifndef GEMM_TYPES_TD
#define GEMM_TYPES_TD
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/Traits.td"

class GemmC_Type<string name,string typememonic,list<Trait> traits = []> : 
        TypeDef<GemmCDialect,name,traits>{
    let mnemonic = typememonic;
}

def GemmC_Int4 : I<4>;
def GemmC_Int8 : I<8>;
def GemmC_Int16 : I<16>;
def GemmC_Int32 : I<32>;
def GemmC_Int64 : I<64>;
def GemmC_Float : F<32>;
def GemmC_AnyNum:AnyTypeOf<[GemmC_Int4, GemmC_Int8, GemmC_Int16, GemmC_Int32, GemmC_Int64, GemmC_Float]>;

def HasNo0Dim : And<[IsRankedTensorTypePred,
                CPred<"::llvm::all_of(::llvm::cast<::mlir::RankedTensorType>($_self).getShape(),[](auto v){return v!=0;})">]>;

def AtLeastRankOne : And<[IsRankedTensorTypePred,
                CPred<"::llvm::cast<::mlir::RankedTensorType>($_self).getRank()>=1">]>;


class GemmCTensorof<list<Type> allowedTypes,string summary = "GemmC-conformat tensor">
        : TensorOf<allowedTypes, [Or<[HasNo0Dim, AtLeastRankOne]>],summary>;

def GemmCTensor : GemmCTensorof<[GemmC_AnyNum]>;

//Attributes
class GemmC_Attr<string name, string attrmnemonic, list<Trait> traits = []> : AttrDef<GemmCDialect, name, traits>{
    let mnemonic = attrmnemonic;
}

def TransposeAttr : GemmC_Attr<"trans", "trans">{
    let summary = "Transpose attribute";
    let parameters = (ins GemmC_Int8:$trans);
    let assemblyFormat = "`<`$trans`>`";
}

#endif
