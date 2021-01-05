FROM petronetto/pytorch-alpine
RUN pip install flask
COPY run-blender.sh /golem/entrypoints/
COPY run1.sh /golem/entrypoints/
COPY runold.sh /golem/entrypoints/
COPY runold1.sh /golem/entrypoints/
COPY test.py /golem/entrypoints/
#COPY aesthetic1 /golem/entrypoints/aesthetic1
COPY aethestics /golem/entrypoints/aethestics
RUN chmod g+rx,o+rx /golem/entrypoints/run-blender.sh
RUN chmod g+rx,o+rx /golem/entrypoints/run1.sh
RUN chmod g+rx,o+rx /golem/entrypoints/runold.sh
RUN chmod g+rx,o+rx /golem/entrypoints/runold1.sh
VOLUME /golem/work /golem/output /golem/resource