Dir.mkdir "dat" unless File.exist? "dat"

stride = 16
error_check = 0

[0].each do |order|
  [0,1].each do |transa|
    [0,1].each do |transb|
      5.times do
        ["dgemm","sgemm"].each do |prog|
          #outfile = "dat/mkl1.10.319_#{`hostname`.chop}_#{prog}_"
          outfile = "dat/mkl2013.1.117_#{`hostname`.chop}_#{prog}_"
          #outfile = "dat/clAmdBlas1.7.257_#{`hostname`.chop}_#{prog}_"
          outfile += (order == 0) ? "C" : "R"
          outfile += (transa == 0) ? "N" : "T"
          outfile += (transb == 0) ? "N" : "T"
          outfile += ".txt"
          puts outfile
          #max_size = (prog == "dgemm") ? 5120 : 5120
          max_size = (prog == "dgemm") ? 6192 : 6192
          system "./#{prog} #{order} #{transa} #{transb} #{max_size} #{stride} #{error_check} | tee -a #{outfile}"
        end
      end
    end
  end
end

