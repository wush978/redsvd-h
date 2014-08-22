#include <memory>
#include <Rcpp.h>
#include <RcppEigen.h>
#include <redsvd.h>

using namespace Rcpp;

typedef Eigen::MappedSparseMatrix<double> MSpMat;
typedef Eigen::Map<Eigen::MatrixXd> MapMatd;

template<typename Mat>
SEXP RedSVD_RAPI(SEXP Rm, int rank = 0) {
  typedef RedSVD::RedSVD<Mat> T;  
  std::auto_ptr<T> pretval(NULL);
  Mat m(as< Mat >(Rm));
  if (rank == 0) {
    pretval.reset(new T(m));
  }
  else {
    pretval.reset(new T(m, rank));
  }
  return List::create(
    Named("u") = pretval->matrixU(),
    Named("d") = pretval->singularValues(),
    Named("v") = pretval->matrixV()
    );
}

// TODO: THE_BRACKET_OPERATOR_IS_ONLY_FOR_VECTORS__USE_THE_PARENTHESIS_OPERATOR_INSTEA
//'@export
//[[Rcpp::export("RedSVD.matrix")]]
SEXP RedSVD_matrix(SEXP Rm, int rank = 0) {
  return RedSVD_RAPI<MapMatd>(Rm, rank);
}

//'@export
//[[Rcpp::export("RedSVD.dgCMatrix")]]
SEXP RedSVD_dgCMatrix(SEXP Rm, int rank = 0) {
  return RedSVD_RAPI<MSpMat>(Rm, rank);
}
