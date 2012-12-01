Dir.mkdir "dat" unless File.exist? "dat"

order = 0
stride = 16
error_check = 0

5.times do
  [0,1].each do |transa|
    [0,1].each do |transb|
      ["sgemm","dgemm"].each do |prog|
    #["sgemm"].each do |prog|
        outfile = "dat/clAmdBlas1.8.291tune_#{`hostname`.chop}_#{prog}_"
        #outfile = "dat/clAmdBlas1.7.257_#{`hostname`.chop}_#{prog}_"
        outfile += (order == 0) ? "C" : "R"
        outfile += (transa == 0) ? "N" : "T"
        outfile += (transb == 0) ? "N" : "T"
        outfile += ".txt"
        puts outfile
        max_size = (prog == "dgemm") ? 9000 : 13000
        #max_size = (prog == "dgemm") ? 6784 : 8192
        #max_size = (prog == "dgemm") ? 5700 : 8192
        #max_size = (prog == "dgemm") ? 7200 : 8200
        system "./#{prog} #{order} #{transa} #{transb} #{max_size} #{stride} #{error_check} | tee -a #{outfile}"
      end
    end
  end
end

