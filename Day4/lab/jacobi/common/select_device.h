#ifdef _OPENACC
   int mygpu, myrealgpu, num_devices;
   acc_device_t my_device_type;

   my_device_type = acc_device_nvidia;

   if(argc == 1) mygpu = 0; else mygpu = atoi(argv[1]);

   acc_set_device_type(my_device_type) ;
   num_devices = acc_get_num_devices(my_device_type) ;
   fprintf(stderr,"Number of devices available: %d \n",num_devices);
   acc_set_device_num(mygpu,my_device_type);
   fprintf(stderr,"Trying to use GPU: %d \n",mygpu);
   myrealgpu = acc_get_device_num(my_device_type);
   fprintf(stderr,"Actually I am using GPU: %d \n",myrealgpu);
   if(mygpu != myrealgpu) {
     fprintf(stderr,"I cannot use the requested GPU: %d\n",mygpu);
     exit(1);
   }
#endif
