#version 430

layout(local_size_x = 256) in;


// Fourier Feature Network: 4D input -> 4D output
struct FourierCenter {
    vec4 frequency;  // 4D frequency vector
    vec4 amplitude;  // 4D amplitude/weight vector
};
//10 FourierCenters makes a Rule
struct Rule {
    FourierCenter centers[10];
};
//uniform Rule CURRENT_RULE_UNIFORM;
#define CURRENT_RULE_UNIFORM MY_RULE
const Rule MY_RULE = Rule(FourierCenter[10](
    FourierCenter(
        vec4(-0.1990494728088379, 0.5282548666000366, -0.07602667808532715, 0.5671815872192383),
        vec4(-0.3482027053833008, -0.3454299569129944, 0.7230108976364136, 0.4688270092010498)
    ),
    FourierCenter(
        vec4(0.663841724395752, 0.8687338829040527, 1.343493938446045, -1.5923092365264893),
        vec4(0.778617262840271, -0.8512462973594666, 0.2766317129135132, 0.5831986665725708)
    ),
    FourierCenter(
        vec4(-0.6678104400634766, -0.6573953628540039, 0.3541222810745239, -0.2350684404373169),
        vec4(-0.5221054553985596, -0.40160465240478516, 0.1863727569580078, -0.6424355506896973)
    ),
    FourierCenter(
        vec4(0.8509891033172607, -0.2792264223098755, -1.4230021238327026, 1.1943459510803223),
        vec4(-0.5740256309509277, -0.22188138961791992, 0.6526217460632324, -0.6875842809677124)
    ),
    FourierCenter(
        vec4(2.2007670402526855, -1.2524710893630981, -2.4615631103515625, -1.5341962575912476),
        vec4(0.5721189975738525, -0.7936639785766602, -0.9886353015899658, -0.1873658299446106)
    ),
    FourierCenter(
        vec4(0.43340039253234863, 1.3021230697631836, -1.0023174285888672, 0.8576390743255615),
        vec4(-0.6375958323478699, 0.23635947704315186, 0.7200241088867188, 0.5822278261184692)
    ),
    FourierCenter(
        vec4(-0.852695643901825, -0.2363133430480957, 0.2537001371383667, 0.640865683555603),
        vec4(-0.8523270487785339, -0.37935876846313477, -0.4449183940887451, -0.8622293472290039)
    ),
    FourierCenter(
        vec4(-0.9736353754997253, -0.10914736986160278, 0.5158311128616333, -0.6797989010810852),
        vec4(-0.3303564786911011, 0.8944520950317383, 0.9284071922302246, -0.2780998945236206)
    ),
    FourierCenter(
        vec4(0.8407406806945801, -1.7259242534637451, 0.24772119522094727, -1.989144206047058),
        vec4(0.9510786533355713, 0.9957772493362427, 0.797810435295105, -0.15748149156570435)
    ),
    FourierCenter(
        vec4(-0.6159260869026184, 0.39494407176971436, 0.24917232990264893, -0.7255998849868774),
        vec4(-0.951838493347168, -0.39515429735183716, -0.9195523262023926, 0.6620248556137085)
    )
));

#define COHORTS 64
#define HAZARD_RATE 0.
#define SQRT_WORLD_SIZE 1
#define SENSOR_ANGLE .45
#define SENSOR_DISTANCE .6717398166656

#define RULE_SEED  0
#define MUTATION_SCALE 0
#define SENSOR_GAIN 1.136
#define AXIAL_FORCE .371
#define LATERAL_FORCE -.707
#define GLOBAL_FORCE_MULT .4514051973
#define DRAG -.3501829802989
#define STRAFE_POWER .4199690222
#define PI 3.1415926

uniform sampler2D canvas_texture;
uniform int frame_count;

struct Entity {
    vec2 pos;
    vec2 vel;
    float size;
    float cohort;
    float padding[2];
    vec4 color;
};  // Total: 48 bytes (12 floats)

layout(std430, binding = 0) buffer EntityBuffer {
    Entity entities[];
};

//------------------------------------RANDOM / HASH / NOISE--------------------------------
// PCG hash - bit-exact across all platforms
uint pcg_hash(uint seed) {
    uint state = seed * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float hash(vec2 co){
    uvec2 u = uvec2(floatBitsToUint(co.x), floatBitsToUint(co.y));
    uint h = pcg_hash(u.x ^ pcg_hash(u.y));
    return float(h) / float(0xffffffffu);
}

vec4 hash4(vec2 co){
    return vec4(
        hash(co),
        hash(co*-1+5),
        hash(co.yx-100),
        hash(co.yx*-1 + 25)
    );
}

// Fourier basis evaluation
// This is how the entities evaluate their Rule
vec4 fourier_noise(FourierCenter[10] centers, vec4 signals) {
    vec4 result = vec4(0.0);

    for(int i = 0; i < 10; i++) {
        // Compute phase from dot product of input with frequency vector
        float phase = dot(signals, centers[i].frequency);

        // Add per-center phase offset to break degeneracy at origin
        // Use a deterministic offset based on center index and amplitude values
        float phase_offset =2*float(i) * 0.6283 + centers[i].amplitude.w * 3.14159;

        // Create basis functions from phase with offset
        // Using sin/cos pairs at fundamental and first harmonic for richer representation
        vec4 basis = vec4(
            sin(phase + phase_offset),
            cos(phase + phase_offset * 0.7),  // Different offsets for variety
            sin(phase * 2.0 + phase_offset * 1.3),
            cos(phase * 2.0 + phase_offset * 0.5)
        );

        // Weight and accumulate
        result += centers[i].amplitude * basis;
    }

    return result;
}

//Given a seed, return 10 random FoureierCenters: enough for a Rule.
FourierCenter[10] generate_random_centers(float seed) {
    FourierCenter[10] centers;

    for(int i = 0; i < 10; i++) {
        // Generate frequency vectors
        // Bias towards lower frequencies for smoother base behaviors
        // Range: [-2, 2] with bias towards [-1, 1]
        float freq_scale = 1.0 + 2.0 * pow(hash(vec2(seed, float(i * 8 + 0))), 2.0);
        centers[i].frequency.x = (hash(vec2(seed, float(i * 8 + 0))) * 2.0 - 1.0) * freq_scale;
        centers[i].frequency.y = (hash(vec2(seed, float(i * 8 + 1))) * 2.0 - 1.0) * freq_scale;
        centers[i].frequency.z = (hash(vec2(seed, float(i * 8 + 2))) * 2.0 - 1.0) * freq_scale;
        centers[i].frequency.w = (hash(vec2(seed, float(i * 8 + 3))) * 2.0 - 1.0) * freq_scale;

        // Generate amplitude vectors
        // Range: [-1, 1]
        centers[i].amplitude.x = hash(vec2(seed, float(i * 8 + 4))) * 2.0 - 1.0;
        centers[i].amplitude.y = hash(vec2(seed, float(i * 8 + 5))) * 2.0 - 1.0;
        centers[i].amplitude.z = hash(vec2(seed, float(i * 8 + 6))) * 2.0 - 1.0;
        centers[i].amplitude.w = hash(vec2(seed, float(i * 8 + 7))) * 2.0 - 1.0;
    }

    return centers;
}

vec4 random_fourier_noise(vec4 pos, float seed) {
    FourierCenter[10] centers = generate_random_centers(seed);
    return fourier_noise(centers, pos);
}

vec4 normalized_fourier_noise(vec4 pos, float seed) {
    vec4 noise = random_fourier_noise(pos, seed);
    return noise * 0.1 + 0.5;
}

//------------------------------------RANDOM / HASH / NOISE--------------------------------

//rotate p around origin by angle a
void pR(inout vec2 p, float a) {
	p = cos(a)*p + sin(a)*vec2(p.y, -p.x);
}

//convert p from worldspace to texture coords and retrieve canvas
vec4 get_can(vec2 p){
    vec2 res=textureSize(canvas_texture,0);
    vec2 aspect=vec2(1,res.x/res.y);
    vec2 uv = p/2*aspect+.5;
    return texture(canvas_texture, uv);
}

//normalize vector that tolerates vec2(0)
vec2 safenorm(vec2 p){
    return length(p)==0?vec2(0):normalize(p);
}

//simply assigns each to a cohort based on its index. 
//floor(get_cohort(index)) should be used for cohort equality tests
float get_cohort(uint index) {
    return float(COHORTS) * float(index) / float(entities.length());
}

//Return all entities to their initialization state
void reset(uint index){

    float size=index<entities.length()?.0015: 0;
    float cohort_val = get_cohort(index);

    //set pos and vel to random values on a small disk
    vec2 pos=vec2(hash(vec2(cohort_val, 1.0)), hash(vec2(cohort_val, 2.0))) * 2.0 - 1.0;
    vec2 vel=0.01*.005*(vec2(hash(vec2(cohort_val,index)),hash(vec2(cohort_val,pos.y)))*2-1);
    vec4 color=vec4(0,0,1,.045);
    //store to persistent entity buffer
    entities[index]=Entity(pos,vel,size,cohort_val/float(COHORTS),float[2](0,0),color);
}

//randomly change noise function parameters, scaled by parameter amount. 
//Each cohort gets a unique mutation for any given rule
void mutate_rule(inout Rule current_rule,float amount,float cohort){
    float seed = hash(current_rule.centers[4].frequency.xy+current_rule.centers[7].amplitude.ys+current_rule.centers[1].frequency.zw)+cohort;

    for(int i = 0; i < 10; i++) {
        vec4 amp_mutation = amount * (-1.0 + 2.0 * hash4(-.5+vec2(-i+seed,i)));
        current_rule.centers[i].amplitude += amp_mutation;
        current_rule.centers[i].frequency *= 1 + amount * 0.5 * (hash(vec2(seed,i))-.5);
    }
}


//Used to enforce left-right symmetry in the local coordinates vec2(forward, left)
vec2 y_reflect(vec2 p){
    return p*vec2(1,-1);
}


//Somewhat arbitrary generator of functions with 4 float inputs and 4 float outputs,
//varying rule should smoothly change the behavior of black box. Here, we use fourier noise
vec4 black_box(vec2 L,vec2 R,Rule rule){
    return (fourier_noise(rule.centers, vec4(L,R)));
}



//This function determines entity output by plugging sensor values into a noise function called black_box()
//The calculation is performed twice, once in mirrored coordinates, and the two values are averaged.
//This keeps entities from displaying clockwise/counterclockwise bias.
//PARAMETERS:
//--L and R: velocity field measurements from left sensor and right sensor.
//--axis: forward vector that defines our orientation.
//--rule: coefficients for the noise function that dictates entity behavior.
//RETURNS:
//--force: A "push" vector that will be added to entity.vel
//--strafe: A "hop" vector that will be added to entity.pos and have no effect on velocity
//--color: vec2 to be used as parameters in a coloring function
void calculate_entity_behavior( vec2 L,vec2 R, vec2 axis, Rule rule, out vec2 force, out vec2 strafe, out vec2 color){

    //build a local coordinate frame where "axis" is forward.
    vec2 forward = safenorm(axis);
    vec2 left = vec2(forward.y,-forward.x);

    //Convert L and R to local coordinates.
    //Ie. decompose each into an axial component and a lateral component
    L = vec2(dot(L,forward),dot(L,left));
    R = vec2(dot(R,forward),dot(R,left));

    //calculate black box noise values
    vec4 baseterm = black_box(L,R,rule);
    vec4 mirrorterm = black_box(y_reflect(R),y_reflect(L),rule);

    //Combine base and mirror terms to cancel bias
    force = baseterm.xy + y_reflect(mirrorterm.xy);
    strafe = baseterm.zw + y_reflect(mirrorterm.zw);

    //Convert force and strafe back to world coordinates
    force = (forward * force.x * AXIAL_FORCE) + (left * force.y * LATERAL_FORCE);
    strafe = (forward * strafe.x * AXIAL_FORCE) + (left * strafe.y * LATERAL_FORCE);

    color = baseterm.xy+(mirrorterm.xy); //Just an arbitrary function of blackbox output. Reuses force terms.
    return;
}

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= entities.length()) return;

    Entity e=entities[index];
    float cohort = get_cohort(index);
    Rule current_rule = CURRENT_RULE_UNIFORM;
    //Hazard Rate == probability each frame to reset this particle
    bool hazard_reset = HAZARD_RATE > hash(vec2(float(index)/float(entities.length()),frame_count));
    
    //frame_count == 0 signals a simulation reset
    if (frame_count==0||hazard_reset){reset(index);return;}

    //Calculate position offsets for the two sensors.
    float sample_dist = 1./SQRT_WORLD_SIZE*.005 * SENSOR_DISTANCE;
    vec2 orientation = safenorm(e.vel);//vector facing the same direction as velocity, with length==sample_dist

    vec2 left_sensor_offset = orientation*sample_dist;
    vec2 right_sensor_offset = orientation*sample_dist;
    pR(left_sensor_offset,SENSOR_ANGLE*PI);//rotate them opposite directions
    pR(right_sensor_offset,-SENSOR_ANGLE*PI);

    //read the trails from canvas
    vec4 ltap = get_can(e.pos+left_sensor_offset);
    vec4 rtap = get_can(e.pos+right_sensor_offset);

    //if a few arbitrary coefficients are exactly 0, then assume target_rule is all 0s (no target) and generate a random rule instead.
    if(current_rule.centers[0].frequency==vec4(0) && current_rule.centers[5].amplitude==vec4(0)){
        current_rule = Rule(generate_random_centers(RULE_SEED+floor(cohort)));
    }
    //Each cohort gets a random mutation
    mutate_rule(current_rule,MUTATION_SCALE,RULE_SEED+floor(cohort));

    //rescale sensor values
    float sensor_scaling = SQRT_WORLD_SIZE*38.855*SENSOR_GAIN;
    ltap *= sensor_scaling;
    rtap *= sensor_scaling;

    //compute entity action
    vec2 strafe =vec2(0);//set by calculate_...
    vec2 force = vec2(0);//set by calculate_...
    vec2 col_params = vec2(0);//set by calculate_...
    calculate_entity_behavior(ltap.xy,rtap.xy,orientation,current_rule,force,strafe,col_params);

    //rescale output forces
    force *= 1./SQRT_WORLD_SIZE*GLOBAL_FORCE_MULT/400.;
    strafe *= 1./SQRT_WORLD_SIZE*GLOBAL_FORCE_MULT/20.;


    //e.color is interpreted as vec4(hue,saturation,brightness,alpha)
    //We just set brightness to 1 and modulate hue and saturation


    e.color.x = hash(vec2(floor(cohort))); //just assign a random hue to each cohort
    e.color.y = sin(col_params.y)/2.+.5;//saturation must be 0..1
    e.color.z=1;//brightness 1.
    e.color.w=0.045; //low alpha

    //Accelerate: Apply drag and add force to e.vel,
    e.vel = e.vel*DRAG + force;
    
    //Move: add e.vel and strafe to e.pos
    e.pos += e.vel;
    e.pos += strafe*STRAFE_POWER;

    //wrap from from -1 to 1
    e.pos = 2*(fract(e.pos/2-.5)-.5);

    //Commit new entity state to buffers
    entities[index]=e;
}
