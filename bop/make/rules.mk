MPI_TOK=MPI_SPECIFIC_

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCFLAGS) $< -c -o $@

$(MPI_TOK)%.o: %.cpp
	$(MPICXX) $(MPICXXFLAGS) $(INCFLAGS) -c $< -o $@

%: %.o; $(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)

%: $(MPI_TOK)%.o; $(MPICXX) $(MPICXXFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS_MPI)
