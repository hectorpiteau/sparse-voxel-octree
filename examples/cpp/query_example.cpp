#include <svo/Octree.hpp>
#include <svo/Version.hpp>

#include <iostream>

int main() {
  svo::Octree tree;
  tree.validate();

  std::cout << "svo " << svo::version() << " nodes=" << tree.num_nodes() << '\n';
  return 0;
}
