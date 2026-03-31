  SmallVector<TPSingleDefRecipe *, 32> Worklist;

  ReversePostOrderTraversal<TPBlockDeepTraversalWrapper<TPBlockBase *>>
      RPOT(TPBlockDeepTraversalWrapper<TPBlockBase *>(tplan.get()->getEntry()));
  // for (TPBasicBlock *TPB :
  //      TPBlockUtils::blocksOnly<TPBasicBlock>(RPOT)) {
  for (TPBasicBlock *TPB : TPBlockUtils::blocksOnly<TPBasicBlock>(RPOT)) {
    auto *TPBB = dyn_cast<TPBasicBlock>(TPB);
    if (!TPBB) continue;
    
    // YYG::REMOVE
    errs() << "[RPOT] TPBB: \n";
    TPBB->dump();

    // From outer -> inner
    for (TPRecipeBase &BaseR : *TPBB) {
      // TPSingleDefRecipe 로의 다운캐스팅이 성공했을 때만 처리
      if (auto *Recipe = dyn_cast<TPSingleDefRecipe>(&BaseR)) {
        unsigned Dim = Recipe->getDimIndex();
        // Pass through when Dim is unset(0).
        if (!Dim)
          continue;
        
        // DimSet is a SmallBitVector - It stores bits indexed from 0 to size-1. 
        // To set bit Dim, the vector must have at least Dim + 1 bits.
        Recipe->DimSet.resize(std::max(Recipe->DimSet.size(), (size_t)(Dim + 1)));
        Recipe->DimSet.set(Dim);

        Worklist.push_back(Recipe);
      }
    }
  }
  // Phase 2: BFS union propagation to fixpoint.
  while (!Worklist.empty()) {
    TPSingleDefRecipe *V = Worklist.pop_back_val();
    auto *VRecipe = dyn_cast<TPRecipeBase>(V);
    for (TPUser *U : V->users()) {
      auto *Recipe = dyn_cast<TPRecipeBase>(U);
      if (!Recipe)
        continue;
      
      TPSingleDefRecipe *DV = Recipe->getDefinedValue();
      if (!DV)
        continue;
      auto *DVRecipe = dyn_cast<TPRecipeBase>(DV);
      
      unsigned NeedSize = V->DimSet.size();
      // YYG::REMOVE
      errs() << "V: ";
      VRecipe->dump();
      errs() << "V's size: " << NeedSize << "\n";
      errs() << "U: ";
      Recipe->dump();
      errs() << "DV: ";
      DVRecipe->dump();
      errs() << "DV's size: " << DV->DimSet.size() << "\n";

      if (DV->DimSet.size() < NeedSize)
        DV->DimSet.resize(NeedSize);
      SmallBitVector Before = DV->DimSet;
      DV->DimSet |= V->DimSet;
      if (DV->DimSet != Before)
        Worklist.push_back(DV);
    }
  }
