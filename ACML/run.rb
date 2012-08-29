Dir.mkdir "dat" unless File.exist? "dat"

order = 0
transa = 0
transb = 0
stride = 1
error_check = 0

5.times do
  ["dgemm","sgemm"].each do |prog|
    outfile = "dat/acml5.1.0gfortran_#{`hostname`.chop}_#{prog}_"
    #outfile = "dat/clAmdBlas1.7.257_#{`hostname`.chop}_#{prog}_"
    outfile += (order == 0) ? "C" : "R"
    outfile += (transa == 0) ? "N" : "T"
    outfile += (transb == 0) ? "N" : "T"
    outfile += ".txt"
    puts outfile
    max_size = (prog == "dgemm") ? 5120 : 5120
    system "./#{prog} #{order} #{transa} #{transb} #{max_size} #{stride} #{error_check} | tee -a #{outfile}"
  end
end

