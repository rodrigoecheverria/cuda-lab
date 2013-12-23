#include <iostream>
#include <math.h>
#include <algorithm>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/adjacent_difference.h>
#include <thrust/generate.h>
#include <thrust/unique.h>
#include <thrust/scan.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform.h>
#include <thrust/binary_search.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>

#define SITES            10
#define MAX_MEASUREMENT   8

unsigned int TotalRain ( thrust::device_vector<unsigned int>& M)  
{ return thrust::reduce(M.begin(), M.end()); }

unsigned int TotalDaysRainInSite ( thrust::device_vector<unsigned int>& S, 
                                   const unsigned int Site)
{ return thrust::count(S.begin(), S.end(), Site); }

unsigned int TotalSites ( thrust::device_vector<unsigned int>& S)
{ 
  thrust::pair<thrust::device_vector<unsigned int>::iterator, thrust::device_vector<unsigned int>::iterator> new_end;
  thrust::device_vector<unsigned int> G(S.size());
  thrust::device_vector<unsigned int> D(S.size());
  thrust::device_vector<unsigned int> K(S.size());
  thrust::sort(S.begin(), S.end());
  
  new_end = thrust::reduce_by_key(S.begin(), S.end(), G.begin(),K.begin(), D.begin() );

  return new_end.first - K.begin();
}


typedef thrust::device_vector<int>::iterator   SiteIt;
typedef thrust::device_vector<int>::iterator   MeasureIt;
typedef thrust::tuple<SiteIt, MeasureIt> IteratorTuple;
typedef thrust::zip_iterator<IteratorTuple> ZipIterator;

struct zero_if_not_site : thrust::unary_function<thrust::tuple<int,int>,thrust::tuple<int,int> >
{
    const unsigned int site;
    zero_if_not_site(unsigned int _site) : site(_site) {}
    
    __host__ __device__ thrust::tuple<int,int> operator()(const thrust::tuple<int,int> &x) const
    {
      return thrust::get<0>(x) == site ? x : thrust::make_tuple(thrust::get<0>(x),0);
    }
};
struct add_tuple_value : thrust::binary_function<thrust::tuple<int,int>,thrust::tuple<int,int>,int>
{
  int operator()(const thrust::tuple<int,int> x, const thrust::tuple<int,int> y)
  {
     return thrust::get<1>(x) + thrust::get<1>(y);
  }
  
};
unsigned int TotalRainIN ( thrust::device_vector<unsigned int>& S, 
                           thrust::device_vector<unsigned int>& M, 
                           const unsigned int St)
  {  
    //ZipIterator iter(thrust::make_tuple(S.begin(), M.begin()));
    //ZipIterator result = thrust::partition(iter, iter.end(), in_site(St)); //see
    /*return thrust::transform_reduce(thrust::make_zip_iterator(thrust::make_tuple(S.begin(), M.begin())),
                                    thrust::make_zip_iterator(thrust::make_tuple(S.end(),   M.end  ())),
                                    zero_if_not_site(St),0,
                                    add_tuple_value()
                                    );*/
  return thrust::reduce(thrust::make_zip_iterator(thrust::make_tuple(S.begin(), M.begin())),
                        thrust::make_zip_iterator(thrust::make_tuple(S.end(),   M.end  ())),
                        (unsigned int) 0,
                        add_tuple_value()
                                    );
                                    //return 0;
  }

unsigned int TotalRainBetween ( thrust::device_vector<unsigned int>& D, 
                                thrust::device_vector<unsigned int>& M, 
                                const unsigned int Start, const unsigned int End)
  { return 0; }

unsigned int TotalDaysWithRain ( thrust::device_vector<unsigned int>& D) { return 0; }

unsigned int TotalDaysRainHigher( thrust::device_vector<unsigned int>& D, 
                                  thrust::device_vector<unsigned int>& M, 
                                  const unsigned int Min)
  { return 0;}//thrust::count_if() }


bool Option ( char o, thrust::device_vector<unsigned int>& Days, 
                      thrust::device_vector<unsigned int>& Sites, 
                      thrust::device_vector<unsigned int>& Measurements) 
{
  switch (o) {
    case '0': std::cout << "Total Rainfall is " << TotalRain( Measurements ) << std::endl; break;

    case '1': std::cout << "Total number of Days with any Rainfall in Site 3: " 
                 << TotalDaysRainInSite ( Sites, 3 ) << std::endl;   break;

    case '2': std::cout << "Total Sites with rain: " << TotalSites ( Sites ) << std::endl; break;

    case '3': std::cout << "Total Rainfall in Site 7 is " << TotalRainIN ( Sites, Measurements, 7 )
                 << std::endl; break;

    case '4': std::cout << "Total Rainfall between days 7 and 77 is " 
                 << TotalRainBetween ( Days, Measurements, 7, 77 ) << std::endl; break;

    case '5': std::cout << "Total number of Days with any rainfall: " 
                << TotalDaysWithRain ( Days ) << std::endl;  break;

    case '6': std::cout << "Number of Days where Rainfall exceeded 10 is " 
                << TotalDaysRainHigher ( Days, Measurements, 10 ) << std::endl; break;

    default:  return false;
  }
  return true;
}

struct rand_modulus {
    unsigned int N;
    rand_modulus(unsigned int _NN) : N(_NN) {}

    __host__ __device__
        unsigned int operator()() const { 
            return rand() % N; //N*SITES
        }
};

struct is_equal {
    __host__ __device__
        unsigned int operator() ( const unsigned int& d, const unsigned int& s )  { 
            return d==s? 1: 0;
        }
};

struct get_site {
    __host__ __device__
        unsigned int operator() ( const unsigned int& v )  { 
            return v % SITES;
        }
};

struct get_day {
    __host__ __device__
        unsigned int operator() ( const unsigned int& v )  { 
            return v / SITES;
        }
};


unsigned int rand_mes() {
  return  (unsigned int) pow( 2.0, ((double) (rand() % 100000)) /  (100000 / MAX_MEASUREMENT) );
}


int main (int argc, char **argv)
{
  unsigned int N=20;
  char o= '1';
  int Dup = -1;

  if (argc>1) {  o = argv[1][0];  }
  if (argc>2) {  N = atoi(argv[2]); }
  
  if (o == 'H' || o == 'h') {
    std::cout <<  "Arguments: (H|1|2|3|4|5|6) N " << std::endl;
    exit(0);
  }

  // use this host vector to generate random input data
  thrust::host_vector<unsigned int> HDay(N);
  thrust::host_vector<unsigned int> HMes(N);

  srand(0); // init random generation seed: same random numbers generated in each execution

  // Generate Information sorted by (increasing) day and site, and with no duplicates (day, site)
  thrust::generate ( HDay.begin(), HDay.end(), rand_modulus(N*SITES) );
  thrust::generate ( HMes.begin(), HMes.end(), rand_mes ); 

  // Create Device vectors and copy data from host vectors
  thrust::device_vector<unsigned int> Days        = HDay;
  thrust::device_vector<unsigned int> Measurements= HMes;
  thrust::device_vector<unsigned int> Sites(N);

  // Sort data and modify to avoid duplicates ( only works fine if SITES=10 )
  thrust::sort ( Days.begin(), Days.end() ); 
  do {
    Dup++;
    thrust::transform ( Days.begin(), Days.end()-1, Days.begin()+1, Sites.begin(), is_equal() );
    thrust::transform ( Days.begin()+1, Days.end(), Sites.begin(), Days.begin()+1, thrust::plus<unsigned int>() ); 
  } while (thrust::reduce ( Sites.begin(), Sites.end()-1 ) > 0);

  thrust::transform ( Days.begin(), Days.end(), Sites.begin(), get_site() );
  thrust::transform ( Days.begin(), Days.end(), Days.begin(), get_day() );


  if (Dup >0)
     std::cout << "Phases to extract duplicates during generation: " << Dup << std::endl << std::endl;

  if ( N<=20 ) { // for small cases: print contains of input vectors
    std::cout << "Days:         ";
    thrust::copy( Days.begin(), Days.end(), std::ostream_iterator<unsigned int>( std::cout, ", " ));
    std::cout << std::endl << "Sites:        "; 
    thrust::copy( Sites.begin(), Sites.end(), std::ostream_iterator<unsigned int>( std::cout, ", " ));
    std::cout << std::endl << "Measurements: "; 
    thrust::copy( HMes.begin(), HMes.end(), std::ostream_iterator<unsigned int>( std::cout, ", " ));
    std::cout << std::endl;
  } 

  // create device vectors and copy data from host vectors

  Option ( o, Days, Sites, Measurements);
 
  return 0;
}
